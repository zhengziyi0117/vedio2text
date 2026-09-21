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
# 登录态：B 站的 CC/AI 字幕、会员清晰度都要 cookie，匿名拿不到（yt-dlp 会警告
# "Subtitles are only available when logged in" 然后降级走 ASR）。值填浏览器名
# （chrome / safari / firefox…）走 --cookies-from-browser，填 cookies.txt 路径走 --cookies。
COOKIES = os.getenv("V2T_COOKIES", "")
# 有现成字幕默认就复用（快，B 站字幕秒级；ASR 98 分钟音频要 8 分钟）。
# 但现成字幕常是机翻/无标点，whisper 转写通常更准 —— 要更准就置 1 强制走 ASR。
FORCE_ASR = os.getenv("V2T_FORCE_ASR", "").strip() not in ("", "0")


def _cookies() -> list:
    if not COOKIES:
        return []
    p = Path(COOKIES).expanduser()
    return ["--cookies", str(p)] if p.exists() else ["--cookies-from-browser", COOKIES]


# 带 list= 的 YouTube 链接默认会拖整个播放列表下来 —— 一个 work 目录只装一讲，必须掐掉
YTDLP = ["yt-dlp", "--no-playlist", *_cookies()]
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


def _sub_files(work: Path) -> list[Path]:
    return list(work.glob("source*.vtt")) + list(work.glob("source*.srt"))


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
    if not _sub_files(work):
        # 第一趟只要首选语种（--sub-langs 是正则 fullmatch，不是 glob：B 站的
        # 机翻码 ai-zh 得写 ai-.*，写成 ai-* 匹配不到且返回码仍是 0，静默一条都不下）。
        # 一条没捞着就退一步抓任意语种 —— 日语课只挂 ja 轨也比回去跑 ASR 强，
        # 文稿那步会按 V2T_DOC_LANG 翻译。danmaku 是弹幕不是字幕，排掉。
        for langs in ("zh.*,ai-.*,en.*", "all,-danmaku"):
            r = run([*YTDLP, "--skip-download", "--write-subs", "--write-auto-subs",
                     "--sub-langs", langs, "--sub-format", "vtt", "--convert-subs", "vtt",
                     *_limit_rate(), "-o", str(work / "source.%(ext)s"), src], check=False)
            if _sub_files(work):
                if langs != "zh.*,ai-.*,en.*":
                    print("[下载] 没有首选语种字幕，退而抓了其它语种（文稿会翻译）")
                break
        if not _sub_files(work):
            # 语种没匹配上时 yt-dlp 返回 0 且一个字幕都不下（stderr 还是空的），
            # 只看返回码会一路静默走 ASR —— 这时那句 "no subtitles for the requested
            # languages" 在 stdout 里。
            last = next((l for l in reversed((r.stderr or r.stdout).strip().splitlines())
                         if l.strip()), "")
            print(f"[下载] 字幕拿不到，走 ASR。原因：{last[:140]}")

    got = _pick_source(work, want_video)
    if not got:
        sys.exit("yt-dlp 没产出可用的媒体文件")
    return got


def playlist_entries(url: str) -> list[dict]:
    """播放列表的分集 [{index, title, url}]；不是播放列表就返回空。

    这里故意不用 YTDLP —— 它带 --no-playlist，会把列表压成单个视频。
    """
    r = run(["yt-dlp", "--flat-playlist", *_cookies(),
             "--print", "%(playlist_index)s\t%(title)s\t%(url)s", url], check=False)
    eps = []
    for line in r.stdout.splitlines():
        idx, _, rest = line.partition("\t")
        title, _, link = rest.partition("\t")
        if idx.strip().isdigit() and link.strip():
            eps.append({"index": int(idx), "title": title, "url": link.strip()})
    return eps


def _base_lang(lang: str) -> str:
    """en-US / en_US → en；zh-Hans、zh-Hans-ar 原样返回。

    yt-dlp 下下来的英文轨叫 en-US，而 LANG_PREF 里只写了 en —— 不归一的话
    en-US 会跟 ab 这种无关语种并列排最后，再按文件名排序就让 source.ab.vtt
    赢了（实测 CS336 第 11、14 讲就是这么用上阿布哈兹语机翻的）。

    只认「主语言-两位大写地区」这一种形式：zh-Hans-ar 是「从 ar 翻译来的
    zh」，内容语言未必是中文，不能当中文轨。
    """
    parts = lang.split("-")
    if len(parts) == 2 and len(parts[1]) == 2 and parts[1].isupper():
        return parts[0]
    return lang


def _lang_rank(p: Path) -> int:
    """source.zh-CN.vtt → 0（最想要）；认不出的语言排最后。

    V2T_LANG 指定了源语言就以它为准：英文课上的 zh.* 是 YouTube 机翻，
    拿机翻当原文再校对，等于白劣化一遍。
    """
    lang = p.stem.split(".", 1)[1] if "." in p.stem else ""
    # B 站的机翻字幕码是 ai-zh / ai-en，去掉前缀才认得出语种，
    # 否则设了 V2T_LANG 时它会被当成「认不出的语种」丢掉，白下载一趟。
    lang = lang.removeprefix("ai-")
    base = _base_lang(lang)
    if LANG and (lang == LANG or lang.split("-")[0] == LANG.split("-")[0]):
        return -1
    # 整串先比，再退到归一后的主语言：zh-Hans 原样命中，en-US 落到 en
    for cand in (lang, base):
        if cand in LANG_PREF:
            return LANG_PREF.index(cand)
    return len(LANG_PREF)


def existing_subs(video: Path, work: Path) -> list[dict] | None:
    """按优先级找现成字幕：yt-dlp 下载的 → 同名外挂 → 内嵌。都没有返回 None。

    V2T_LANG 指定了源语言时它排最前（英文课上的 zh.* 是 YouTube 机翻）；
    但指定语种一条都没有时不会回去跑 ASR —— 拿剩下的最好那条，
    文稿那步会按 V2T_DOC_LANG 翻译过来。
    """
    files = sorted(_sub_files(work), key=lambda p: (_lang_rank(p), p.name))
    for p in files:
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
        # 内嵌字幕轨常是坏的字幕流（尤其 webm/flv 的乱码轨），ffmpeg 抽不出来
        # 不能把整条流程带走 —— 抽失败就当没有，回去跑 ASR。
        out = work / "embedded.srt"
        r = run(["ffmpeg", "-y", "-i", str(video), "-map", "0:s:0", "-c:s", "srt", str(out)],
                check=False)
        if r.returncode == 0 and out.exists():
            segs = parse_subs(out.read_text(errors="ignore"))
            if len(segs) >= 10:
                print(f"[字幕] 用内嵌字幕 {len(segs)} 条")
                return segs
    return None


def extract_audio(video: Path, out: Path):
    print("[音频] ffmpeg 抽取 16k 单声道…")
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(out)])


def _is_apple_silicon() -> bool:
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# whisper.cpp 编译好的 whisper-cli 路径：本机装的是 Vulkan 构建（AMD 核显跑不了 CUDA/ROCm，
# 走 Vulkan 通用后端），非 Apple Silicon 时优先用它，没有再退回纯 CPU 的 faster-whisper。
WHISPER_CPP_BIN = os.getenv("V2T_WHISPER_CPP_BIN") or str(
    ROOT / ".tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
)
WHISPER_CPP_MODEL = os.getenv("V2T_WHISPER_CPP_MODEL") or str(
    ROOT / ".tools" / "whisper.cpp" / "models" / "ggml-large-v3-turbo.bin"
)


def asr(audio: Path) -> list[dict]:
    if _is_apple_silicon():
        return _asr_mlx(audio)
    if Path(WHISPER_CPP_BIN).exists() and Path(WHISPER_CPP_MODEL).exists():
        return _asr_whisper_cpp(audio)
    return _asr_faster_whisper(audio)


def _asr_mlx(audio: Path) -> list[dict]:
    import mlx_whisper
    # models/<名字>/ 优先：huggingface_hub 在本机下 safetensors 会卡死在 0 字节
    # （hf-xet 1.6.0 + huggingface-hub 1.31.0），curl 手动放到这里更可靠。
    local = ROOT / "models" / ASR_MODEL.split("/")[-1]
    repo = str(local) if (local / "config.json").exists() else ASR_MODEL
    print(f"[ASR] mlx-whisper {repo}（走 HF 首次要下 ~1.6GB）…")
    r = mlx_whisper.transcribe(str(audio), path_or_hf_repo=repo,
                               language=LANG, verbose=False)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in r["segments"]]


def _asr_whisper_cpp(audio: Path) -> list[dict]:
    import json
    import tempfile

    print(f"[ASR] whisper.cpp（Vulkan，{WHISPER_CPP_MODEL.split('/')[-1]}）…")
    with tempfile.TemporaryDirectory() as tmp:
        wav16 = Path(tmp) / "audio16k.wav"
        # whisper.cpp 只吃 16k 单声道 wav；audio 已经是 16k wav（extract_audio 产出的），
        # 但直接喂路径更省一次转码，这里留后手用 ffmpeg 兜底非标准输入。
        run(["ffmpeg", "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", str(wav16)])
        out_prefix = Path(tmp) / "out"
        cmd = [WHISPER_CPP_BIN, "-m", WHISPER_CPP_MODEL, "-f", str(wav16),
               "-oj", "--no-prints", "-of", str(out_prefix)]
        if LANG:
            cmd += ["-l", LANG]
        run(cmd)
        data = json.loads((Path(tmp) / "out.json").read_text())
    return [{"start": seg["offsets"]["from"] / 1000, "end": seg["offsets"]["to"] / 1000,
              "text": seg["text"].strip()} for seg in data["transcription"]]


def _asr_faster_whisper(audio: Path) -> list[dict]:
    from faster_whisper import WhisperModel
    # 非 Apple Silicon（Linux/Windows，CPU 或 CUDA）：CTranslate2 后端，不依赖 mlx。
    # mlx 专用的模型名（mlx-community/...）在这里不适用，退回标准 faster-whisper 模型名。
    model_name = os.getenv("V2T_FW_MODEL") or (
        "large-v3-turbo" if "large-v3-turbo" in ASR_MODEL else "large-v3"
    )
    device = os.getenv("V2T_FW_DEVICE", "cpu")
    compute_type = os.getenv("V2T_FW_COMPUTE", "int8" if device == "cpu" else "float16")
    print(f"[ASR] faster-whisper {model_name}（{device}/{compute_type}，首次要下模型）…")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(str(audio), language=LANG, vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def build_subs(video: Path, work: Path) -> list[dict]:
    # 字幕源只影响转写这一步，ASR 是不是更准见 AGENTS.md
    segs = None if FORCE_ASR else existing_subs(video, work)
    if FORCE_ASR and _sub_files(work):
        print("[字幕] V2T_FORCE_ASR=1，无视现成字幕，强制转写")
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
