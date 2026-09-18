"""取材：下载媒体、找现成字幕、没有就转写。

产出 `[{start, end, text}]`，时间戳和文本原样交给 subs 归并。
"""
import os
import re
import subprocess
import sys
from pathlib import Path

from .subs import merge_segments, parse_subs

ASR_MODEL = os.getenv("V2T_ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
LANG = os.getenv("V2T_LANG") or None  # 源语言，None = whisper 自动检测 / 字幕按 LANG_PREF 挑
# 下载限速：跑满带宽容易招来 429（YouTube 字幕接口尤其敏感）。
# 设成 0 或空串即不限速。
RATE_LIMIT = os.getenv("V2T_RATE_LIMIT", "2M")
# 带 list= 的 YouTube 链接默认会拖整个播放列表下来 —— 一个 work 目录只装一讲，必须掐掉
YTDLP = ["yt-dlp", "--no-playlist"]
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".flv", ".mov", ".avi", ".m4a", ".mp3", ".opus", ".wav"}
LANG_PREF = ["zh-Hans", "zh-CN", "zh", "zh-TW", "zh-Hant", "en"]

# v2t/ 的上一层，也就是项目根：models/ 装在这儿
ROOT = Path(__file__).resolve().parent.parent


def run(cmd, check=True):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"命令失败: {' '.join(str(c) for c in cmd)}\n{r.stderr[-2000:]}")
    return r


def _limit_rate() -> list:
    return ["--limit-rate", RATE_LIMIT] if RATE_LIMIT else []


def slugify(s: str) -> str:
    s = re.sub(r"[^\w一-鿿]+", "-", s).strip("-")
    # 按字节限长（文件名单段 255 字节，中文一字 3 字节），不是字符数：
    # 截到 60 字符会把区分各讲的尾部 "Lecture N" 切掉，同一门课的每一讲
    # 都撞进同一个目录，后一讲直接复用前一讲的视频和文稿。
    return s.encode()[:200].decode("utf-8", "ignore") or "course"


def has_video_stream(p: Path) -> bool:
    r = run(["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(p)], check=False)
    return bool(r.stdout.strip())


def _pick_source(work: Path, want_video: bool) -> Path | None:
    """找已下好的媒体文件；want_video 时要求它真的含视频流，别把纯音轨当成视频。"""
    for p in sorted(work.glob("source.*")):
        if p.suffix.lower() in VIDEO_EXT and (not want_video or has_video_stream(p)):
            return p
    return None


def fetch_video(src: str, work: Path, want_video: bool = False) -> Path:
    """本地文件直接返回；URL 走 yt-dlp + 顺手拽字幕。

    want_video 决定下载格式：截图要画面，纯转写只要音轨（体积差约 10 倍）。
    """
    if not src.startswith(("http://", "https://")):
        p = Path(src).expanduser()
        if not p.exists():
            sys.exit(f"文件不存在: {p}")
        return p

    got = _pick_source(work, want_video)
    if got:
        print(f"[下载] 复用 {got.name}")
    else:
        fmt = "bv*[height<=720]+ba/b[height<=720]" if want_video else "ba/b"
        # 视频单独一个文件名：先跑过纯音轨的话 source.webm 已存在，
        # yt-dlp 见了同名文件直接跳过，720p 永远下不来。
        out = work / ("source.video.%(ext)s" if want_video else "source.%(ext)s")
        print(f"[下载] yt-dlp（{'720p 视频' if want_video else '仅音轨'}，限速 {RATE_LIMIT or '关'}）…")
        run([*YTDLP, "-f", fmt, *_limit_rate(), "-o", str(out), src])

    # 字幕是「有则省事、无则 ASR」的降级路径，拿不到不能拖垮主流程。
    # 本地已有字幕文件就不再联网请求一次（YouTube 对这个接口限流很凶）。
    if not (list(work.glob("source*.vtt")) or list(work.glob("source*.srt"))):
        r = run([*YTDLP, "--skip-download", "--write-subs", "--write-auto-subs",
                 "--sub-langs", "zh.*,en.*", "--sub-format", "vtt", "--convert-subs", "vtt",
                 *_limit_rate(), "-o", str(work / "source.%(ext)s"), src], check=False)
        if r.returncode:
            last = next((l for l in reversed(r.stderr.strip().splitlines()) if l.strip()), "")
            print(f"[下载] 字幕拿不到，走 ASR。原因：{last[:140]}")

    got = _pick_source(work, want_video)
    if not got:
        sys.exit("yt-dlp 没产出可用的媒体文件")
    return got


def playlist_entries(url: str) -> list[dict]:
    """播放列表的分集 [{index, title, url}]；不是播放列表就返回空。

    这里故意不用 YTDLP —— 它带 --no-playlist，会把列表压成单个视频。
    """
    r = run(["yt-dlp", "--flat-playlist",
             "--print", "%(playlist_index)s\t%(title)s\t%(url)s", url], check=False)
    eps = []
    for line in r.stdout.splitlines():
        idx, _, rest = line.partition("\t")
        title, _, link = rest.partition("\t")
        if idx.strip().isdigit() and link.strip():
            eps.append({"index": int(idx), "title": title, "url": link.strip()})
    return eps


def _lang_rank(p: Path) -> int:
    """source.zh-CN.vtt → 0（最想要）；认不出的语言排最后。

    V2T_LANG 指定了源语言就以它为准：英文课上的 zh.* 是 YouTube 机翻，
    拿机翻当原文再校对，等于白劣化一遍。
    """
    lang = p.stem.split(".", 1)[1] if "." in p.stem else ""
    if LANG and (lang == LANG or lang.split("-")[0] == LANG.split("-")[0]):
        return -1
    return LANG_PREF.index(lang) if lang in LANG_PREF else len(LANG_PREF)


def existing_subs(video: Path, work: Path) -> list[dict] | None:
    """按优先级找现成字幕：yt-dlp 下载的 → 同名外挂 → 内嵌。都没有返回 None。"""
    files = list(work.glob("source*.vtt")) + list(work.glob("source*.srt"))
    # 指定了源语言就只认那个语种的文件。排序管不着这件事：磁盘上只剩一条
    # zh.* 时它照样排第一，机翻就被当成原文送下去了 —— 宁可不复用，回去跑 ASR。
    if LANG:
        files = [p for p in files if _lang_rank(p) == -1]
    for p in sorted(files, key=lambda p: (_lang_rank(p), p.name)):
        segs = parse_subs(p.read_text(errors="ignore"))
        if len(segs) >= 10:
            print(f"[字幕] 用 yt-dlp 字幕 {p.name}")
            return segs

    base = str(video).rsplit(".", 1)[0]
    for ext in ("srt", "vtt", "ass"):
        p = Path(f"{base}.{ext}")
        if p.exists():
            segs = parse_subs(p.read_text(errors="ignore"))
            if len(segs) >= 10:
                print(f"[字幕] 用外挂字幕 {p.name}")
                return segs

    probe = run(["ffprobe", "-v", "error", "-select_streams", "s",
                 "-show_entries", "stream=index", "-of", "csv=p=0", str(video)], check=False)
    if probe.stdout.strip():
        out = work / "embedded.srt"
        run(["ffmpeg", "-y", "-i", str(video), "-map", "0:s:0", "-c:s", "srt", str(out)])
        segs = parse_subs(out.read_text(errors="ignore"))
        if len(segs) >= 10:
            print(f"[字幕] 用内嵌字幕 {len(segs)} 条")
            return segs
    return None


def extract_audio(video: Path, out: Path):
    print("[音频] ffmpeg 抽取 16k 单声道…")
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(out)])


def asr(audio: Path) -> list[dict]:
    import mlx_whisper
    # models/<名字>/ 优先：huggingface_hub 在本机下 safetensors 会卡死在 0 字节
    # （hf-xet 1.6.0 + huggingface-hub 1.31.0），curl 手动放到这里更可靠。
    local = ROOT / "models" / ASR_MODEL.split("/")[-1]
    repo = str(local) if (local / "config.json").exists() else ASR_MODEL
    print(f"[ASR] {repo}（走 HF 首次要下 ~1.6GB）…")
    r = mlx_whisper.transcribe(str(audio), path_or_hf_repo=repo,
                               language=LANG, verbose=False)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in r["segments"]]


def build_subs(video: Path, work: Path) -> list[dict]:
    segs = existing_subs(video, work)
    if segs:
        print(f"[字幕] 复用已有字幕 {len(segs)} 条 cue")
    else:
        print("[字幕] 无现成字幕 → 转写")
        audio = work / "audio.wav"
        if not audio.exists():
            extract_audio(video, audio)
        segs = asr(audio)
    merged = merge_segments(segs)
    print(f"[字幕] 归并后 {len(merged)} 条")
    return merged
