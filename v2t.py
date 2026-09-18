#!/usr/bin/env python3
"""video-2-text — 课程视频 → 字幕 + 可读文案。

  python v2t.py <本地视频|URL>          # 全流程
  python v2t.py <src> --from clean      # 跳过下载/ASR，只重跑 LLM 校对
  python v2t.py <src> --only subs       # 只出字幕，不写文案
  python v2t.py --selftest              # 解析器自检，不联网

产物在 work/<课程名>/：subs.json(原始) clean.json(校对后) transcript.srt course.md
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# 优先跟随 Claude Code 已配好的 ANTHROPIC_MODEL（走自建网关时通常就是这个），
# 免得脚本自己写死的模型名把用户的配置顶掉、还多花钱
MODEL = os.getenv("V2T_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-opus-5"
ASR_MODEL = os.getenv("V2T_ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
LANG = os.getenv("V2T_LANG") or None  # 源语言，None = whisper 自动检测 / 字幕按 LANG_PREF 挑
DOC_LANG = os.getenv("V2T_DOC_LANG", "中文")  # 文稿写成什么语言
CHUNK = 40          # 每块送 LLM 的字幕条数
WORKERS = 4
STEPS = ["subs", "clean", "doc"]
SCENE_THRESHOLD = 0.4   # 场景切换灵敏度
SHEET = 9               # 每张拼图放几帧（3x3）
SHOT_BATCH = SHEET * 3  # 每批送审的帧数
ASSETS_DIR = "assets"   # 配图目录（相对 course.md）
# 下载限速：跑满带宽容易招来 429（YouTube 字幕接口尤其敏感）。
# 设成 0 或空串即不限速。
RATE_LIMIT = os.getenv("V2T_RATE_LIMIT", "2M")
# 带 list= 的 YouTube 链接默认会拖整个播放列表下来 —— 一个 work 目录只装一讲，必须掐掉
YTDLP = ["yt-dlp", "--no-playlist"]


# ---------------------------------------------------------------- 基础工具

def run(cmd, check=True):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"命令失败: {' '.join(str(c) for c in cmd)}\n{r.stderr[-2000:]}")
    return r


def _limit_rate() -> list:
    return ["--limit-rate", RATE_LIMIT] if RATE_LIMIT else []


def slugify(s: str) -> str:
    s = re.sub(r"[^\w一-鿿]+", "-", s).strip("-")
    return s[:60] or "course"


def _parse_ts(s: str) -> float:
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest.replace(",", "."))


def _fmt_ts(x: float, sep: str = ",") -> str:
    # 先归整到毫秒再拆分，否则 59.9999 会被格式化成非法的 "00:00:60.000"
    h, r = divmod(round(x * 1000), 3_600_000)
    m, r = divmod(r, 60_000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _join(a: str, b: str) -> str:
    """中文直接拼；英文单词间补空格。"""
    if a and b and a[-1].isascii() and a[-1].isalnum() and b[0].isascii() and b[0].isalnum():
        return a + " " + b
    return a + b


# ---------------------------------------------------------------- 字幕解析

TS_RE = re.compile(r"(\d+:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d+:\d{2}:\d{2}[.,]\d{1,3})")
# SRT 序号行不在这里挡：它在 cue 之外，已被 cur is None 的逻辑跳过。
# 若在此匹配 \d+$，会把内容为纯数字的字幕行（"3"）一起吃掉。
SKIP_RE = re.compile(r"^(WEBVTT|NOTE|STYLE|Kind:|Language:)")


def _clean_cue(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)      # <c>, <00:00:01.000>
    s = re.sub(r"\{[^}]*\}", "", s)    # {\an8}
    return s.strip()


def parse_subs(text: str) -> list[dict]:
    """SRT / VTT 都能吃。返回 [{start, end, text}]。"""
    out, cur = [], None
    for line in text.splitlines():
        line = line.rstrip()
        m = TS_RE.search(line)
        if m:
            if cur and cur["lines"]:
                out.append(cur)
            cur = {"start": _parse_ts(m.group(1)), "end": _parse_ts(m.group(2)), "lines": []}
            continue
        if cur is None:
            continue
        if not line:
            if cur["lines"]:
                out.append(cur)
                cur = None
            continue
        if SKIP_RE.match(line):
            continue
        t = _clean_cue(line)
        if t:
            cur["lines"].append(t)
    if cur and cur["lines"]:
        out.append(cur)
    return [{"start": c["start"], "end": c["end"], "text": " ".join(c["lines"]).strip()}
            for c in out if " ".join(c["lines"]).strip()]


def merge_segments(segs, max_gap=1.2, max_len=60, max_dur=14.0):
    """归一成句子级：去重、消滚动字幕、合并碎片。时间戳不动，只并区间。"""
    out = []
    for s in segs:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        if out:
            prev = out[-1]
            # YouTube 滚动字幕：新 cue 是旧 cue 的超集 → 覆盖
            if t.startswith(prev["text"]) and s["end"] >= prev["start"]:
                prev["text"] = t
                prev["end"] = max(prev["end"], s["end"])
                continue
            # 旧 cue 已涵盖新 cue → 只延长
            if prev["text"].endswith(t):
                prev["end"] = max(prev["end"], s["end"])
                continue
            if (t == prev["text"] or s["start"] - prev["end"] < max_gap) and \
               len(prev["text"]) + len(t) <= max_len and s["end"] - prev["start"] <= max_dur:
                prev["text"] = _join(prev["text"], t)
                prev["end"] = max(prev["end"], s["end"])
                continue
        out.append({"start": s["start"], "end": s["end"], "text": t})
    return out


def to_srt(segs) -> str:
    return "\n".join(
        f"{i}\n{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}\n{s['text']}\n"
        for i, s in enumerate(segs, 1)
    )


# ---------------------------------------------------------------- 步骤 1: 取视频

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".flv", ".mov", ".avi", ".m4a", ".mp3", ".opus", ".wav"}


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
        print(f"[下载] yt-dlp（{'720p 视频' if want_video else '仅音轨'}，限速 {RATE_LIMIT or '关'}）…")
        run([*YTDLP, "-f", fmt, *_limit_rate(), "-o", str(work / "source.%(ext)s"), src])

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


LANG_PREF = ["zh-Hans", "zh-CN", "zh", "zh-TW", "zh-Hant", "en"]


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
    local = Path(__file__).resolve().parent / "models" / ASR_MODEL.split("/")[-1]
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


# ---------------------------------------------------------------- 步骤 2: LLM

def llm(system: str, user, max_tokens: int = 16000, think: bool = True) -> str:
    """user 传字符串，或 content blocks 列表（图文混排）。

    有 API key 或 auth token 就直连（ANTHROPIC_BASE_URL 会一并生效，
    所以走自建网关、只配 ANTHROPIC_AUTH_TOKEN 的情况也能用）；
    两者都没有才回退到本机 claude CLI。

    think=False 关掉思考链。校对、列术语这类机械 JSON 任务不需要推理，
    而有些模型（实测 claude-deepseek-v4.1-flash）会一路想到 max_tokens
    耗尽、正文一个字都不吐。
    """
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return _llm_api(system, user, max_tokens, think)
    return _llm_cli(system, user)


def _llm_api(system: str, user, max_tokens: int, think: bool = True) -> str:
    import anthropic
    client = anthropic.Anthropic()
    with client.messages.stream(
        model=MODEL, max_tokens=max_tokens,
        thinking={"type": "adaptive" if think else "disabled"},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as st:
        msg = st.get_final_message()
    if msg.stop_reason == "refusal":
        cat = getattr(msg.stop_details, "category", None)
        sys.exit(f"模型拒绝了请求（{cat}）")
    if msg.stop_reason == "max_tokens":
        sys.exit("输出被 max_tokens 截断，调小 CHUNK 或调大 max_tokens")
    return "".join(b.text for b in msg.content if b.type == "text")


def _llm_cli(system: str, user) -> str:
    """最后的兜底：本机 claude CLI，用你已登录的额度。

    --system-prompt 顶掉它默认的编码 agent 人设，--strict-mcp-config
    少带一堆 MCP 工具定义；模型是 CLI 自己的默认值，V2T_MODEL 在这里不生效。

    ponytail: 每次调用仍会带 Claude Code 的系统提示和内置工具定义（实测
    ~15k token，直连只要几十），能命中它的 prompt cache 所以不算太离谱。
    真嫌贵就配 ANTHROPIC_AUTH_TOKEN 走直连。
    """
    if not isinstance(user, str):
        sys.exit("本地 claude CLI 不支持图文混排，配 ANTHROPIC_API_KEY 或去掉 --shots")
    if not shutil.which("claude"):
        sys.exit("既没有 ANTHROPIC_API_KEY，也找不到 claude 命令。二选一装上。")
    # user 走 stdin：整篇字幕能到几十万字符，塞 argv 会撞 ARG_MAX
    r = subprocess.run(["claude", "-p", "--output-format", "json",
                        "--system-prompt", system, "--strict-mcp-config"],
                       input=user, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"claude CLI 调用失败：{(r.stderr or r.stdout).strip()[-300:]}")
    try:
        return json.loads(r.stdout)["result"]
    except (json.JSONDecodeError, KeyError):
        sys.exit(f"claude CLI 返回不是预期 JSON：{r.stdout[:300]}")


def _json_array(s: str):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j < 0:
        return None
    try:
        v = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, list) else None


GLOSSARY_SYS = """你在为课程字幕校对做准备。根据课程标题和开头文本，列出这门课会出现的专业术语。
输出 JSON 字符串数组，每项格式 "正确写法 ← 常见误听写法"，例如 "反向传播 ← 反相传播"。
只列有把握的，10-30 条。只输出 JSON 数组，不要代码块，不要解释。"""

CLEAN_SYS = """你是课程字幕校对员。输入是 ASR 生成的带编号字幕片段，可能含同音字错误、缺标点、口语赘词、专业术语误听。

输出一个 JSON 字符串数组，长度必须与输入条目数完全一致：第 i 个元素是第 i 条片段的校对结果。

规则：
- 只修字词、标点和术语；绝不合并、拆分、增删条目
- 删掉纯语气词（嗯、啊、呃、那个）和口吃重复
- 该条没有有效内容时输出空字符串
- 保留中英混杂原貌，术语以给定术语表为准
- 只输出 JSON 数组本身，不要 markdown 代码块，不要任何解释"""


def make_glossary(segs, title: str) -> str:
    sample = "\n".join(s["text"] for s in segs[:120])
    arr = _json_array(llm(GLOSSARY_SYS, f"课程标题：{title}\n\n开头文本：\n{sample}", 4000, think=False))
    return "\n".join(str(x) for x in arr) if arr else ""


def _clean_block(block, gloss):
    src = "\n".join(f"{i}. {s['text']}" for i, s in enumerate(block, 1))
    user = (f"术语表：\n{gloss}\n\n" if gloss else "") + \
           f"待校对片段（{len(block)} 条）：\n{src}"
    for _ in range(2):
        arr = _json_array(llm(CLEAN_SYS, user, 8000, think=False))
        if arr and len(arr) == len(block):
            return [str(x or "").strip() for x in arr]
    print(f"  ! 一块校对失败（{len(block)} 条），保留原文", file=sys.stderr)
    return [s["text"] for s in block]


_HALF_PUNCT = {
    ",": "，", ";": "；", ":": "：", "?": "？", "!": "！", ".": "。",
}
# 只动"中文…中文"之间的半角标点，避免误伤 1,000 / 3.14 / e.g.
_CJK_BETWEEN = re.compile(r"(?<=[一-鿿])([,;:?!.])(?=[一-鿿]|$)")


def fix_cjk_punct(s: str) -> str:
    return _CJK_BETWEEN.sub(lambda m: _HALF_PUNCT[m.group(1)], s)


def clean_subs(segs, title: str):
    gloss = make_glossary(segs, title)
    print(f"[校对] 术语表 {len(gloss.splitlines())} 条；{len(segs)} 条字幕分块送模型…")
    blocks = [segs[i:i + CHUNK] for i in range(0, len(segs), CHUNK)]
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        texts = [t for r in ex.map(lambda b: _clean_block(b, gloss), blocks) for t in r]
    out = [{"start": s["start"], "end": s["end"], "text": fix_cjk_punct(t)}
           for s, t in zip(segs, texts) if t]
    print(f"[校对] 保留 {len(out)} 条（丢弃 {len(segs) - len(out)} 条空片段）")
    return out


# ---------------------------------------------------------------- 步骤 3: 文案

DOC_SYS = f"""你在把一门课的完整字幕整理成一篇能替代看视频的文章。

核心要求：**忠实转写，不是提炼**。老师讲了什么就写什么 —— 不归纳、不压缩、
不补充自己的总结或评论。读者要的是"老师讲过的话"，不是你的读书笔记。

- 按老师讲述的顺序组织。**老师一口气连续讲的那一大段内容要合成一个完整段落**，
  不要拆成零散小段，也不要改写成要点清单
- 补上必要的衔接，让段落读起来连贯
- 保留老师的例子、比喻、演示步骤、强调语气和第一人称（"我们来看"、"我演示一下"）
- 保留所有实质内容。宁可长，也不要为了简洁丢信息
- 只去掉真正的口水词和口吃重复（嗯、啊、那个）
- 用 `##` 分小节，标题写具体，别用"第一部分"这种
- 开头一句话说明这堂课讲什么
- 听不清或不确定的内容不要编，宁可不写
- 文稿用{DOC_LANG}写；源字幕是别的语言就翻译过来，专业术语首次出现时括注原文"""

# 配图对齐用：每个自然段回填它的时间，代码据此把截图挂到讲那段的段落后面
DOC_SHOTS_HINT = """

输入的字幕每行带 [MM:SS] 时间标记。在此之上额外要求：
- 在**每一个自然段的末尾**附上该段对应的时间标记 `<!-- MM:SS -->`，
  取输入里出现过的 [MM:SS] 值（该段内容结束处那个），不要自己编。
- 标记独占一行，紧跟在段末，格式严格照抄，不要加别的字。"""


def _mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def make_doc(segs, title: str, want_shots: bool = False) -> str:
    body = "\n".join((f"[{_mmss(s['start'])}] " if want_shots else "") + s["text"]
                     for s in segs)
    print(f"[文案] 全文 {len(body)} 字，送模型…")
    md = llm(DOC_SYS + (DOC_SHOTS_HINT if want_shots else ""),
             f"课程标题：{title}\n\n完整字幕：\n{body}", 48000)
    return fix_cjk_punct(md.strip())


# ---------------------------------------------------------------- 步骤 4: 配图

# 自然段 + 紧随其后的时间标记（由 DOC_SHOTS_HINT 让模型产出）。
# 它既是配图的对齐锚点，最终也会渲染成能跳回视频的时间链接。
# 不用 ^ 锚定行首：段落可以是正文也可以是列表项，只要下一行是标记就算。
PARA_RE = re.compile(r"([^\n]+)\n<!--\s*(\d{1,2}:\d{2})\s*-->")


def _to_sec(mmss: str) -> float:
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def _time_link(sec: float, url: str | None) -> str:
    """段落末尾的时间锚点；有视频地址就做成可点击跳转的链接。"""
    t = _mmss(sec)
    if url and url.startswith("http"):
        sep = "&" if "?" in url else "?"
        return f"[{t}]({url}{sep}t={int(sec)})"
    return t


def extract_frames(video: Path, work: Path, threshold: float = SCENE_THRESHOLD) -> list[dict]:
    """在场景切换处抽帧。返回 [{t, path}] 按时间升序。"""
    out = work / "frames"
    if (work / "frames.txt").exists():
        times = [float(t) for t in (work / "frames.txt").read_text().split()]
        return [{"t": t, "path": p} for t, p in zip(times, sorted(out.glob("*.jpg")))]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[配图] 场景检测抽帧（阈值 {threshold}）…")
    r = run(["ffmpeg", "-v", "info", "-i", str(video),
             "-vf", f"select='gt(scene,{threshold})',showinfo",
             "-fps_mode", "vfr", str(out / "f%04d.jpg")], check=False)
    times = re.findall(r"pts_time:([\d.]+)", r.stderr)
    (work / "frames.txt").write_text("\n".join(times))
    frames = [{"t": t, "path": p} for t, p in zip(times, sorted(out.glob("*.jpg")))]
    if not frames:
        print("[配图] 没抽到场景切换点", file=sys.stderr)
    else:
        print(f"[配图] {len(frames)} 个场景切换点")
    return frames


def make_sheets(cand: list[dict], work: Path, scale: int = 480) -> list[Path]:
    """候选帧拼成 3x3 九宫格送审 —— 逐帧发图的话图片 token 会吃掉整个预算。

    编号规则：从左到右、从上到下，跨图连续（第一张 1-9，第二张 10-18）。
    """
    src, dst = work / "_sheet_in", work / "sheets"
    for d in (src, dst):
        if d.exists():
            shutil.rmtree(d)
    src.mkdir(parents=True)
    dst.mkdir()
    for i, c in enumerate(cand):
        shutil.copy(c["path"], src / f"{i:04d}.jpg")
    n = (len(cand) + SHEET - 1) // SHEET
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src / "%04d.jpg"),
         "-vf", f"scale={scale}:-1,tile=3x3", "-frames:v", str(n),
         str(dst / "sheet%02d.jpg")])
    shutil.rmtree(src)
    return sorted(dst.glob("sheet*.jpg"))


SHOT_SYS = """你在给一篇课程文章挑配图。给你若干张拼图，每张是 3x3 九宫格。
所有格子按「从左到右、从上到下」连续编号：第一张图是 1-9，第二张是 10-18，依此类推。

这些帧是从课程录像里按画面变化自动抽出来的。判断标准要严：**只在"画面本身就是信息"时才配**
—— 图表、示意图、代码、公式、数据表、关键幻灯片，读者看了能拿到正文讲不清楚的细节。

以下一律不配：
- 人物出镜、纯文字封面、过渡动画、空白或模糊
- 只是把正文原话打在屏幕上的字幕式画面，正文自己讲清楚了就不需要图
- 和已选画面重复的（同一张图的不同时刻只留最完整的一张）

宁可少配也不要凑数。

输出 JSON 数组，每项 {"n": 编号, "keep": true/false}，必须覆盖所有编号。
只输出 JSON 数组，不要代码块，不要解释。"""


def _img_block(p: Path) -> dict:
    data = base64.standard_b64encode(p.read_bytes()).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def judge_shots(cand: list[dict], work: Path, title: str) -> set:
    """分批送审（一次看太多帧模型会乱），返回值得配图的 cand 下标集合。"""
    keep = set()
    for base in range(0, len(cand), SHOT_BATCH):
        chunk = cand[base:base + SHOT_BATCH]
        content = [_img_block(s) for s in make_sheets(chunk, work)]
        content.append({"type": "text",
                        "text": f"课程：{title}\n本批 {len(chunk)} 个候选，编号 1-{len(chunk)}。"})
        arr = _json_array(llm(SHOT_SYS, content, 8000))
        if not arr:
            print(f"  ! 第 {base // SHOT_BATCH + 1} 批挑图失败，跳过", file=sys.stderr)
            continue
        for item in arr:
            if not isinstance(item, dict) or not item.get("keep"):
                continue
            try:
                i = int(item["n"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < len(chunk):
                keep.add(base + i)
    return keep


def attach_shots(md: str, frames: list[dict], work: Path, title: str, src: str | None) -> str:
    """按时间把值得配的截图挂到对应段落后，并把时间标记渲染成可跳回视频的链接。

    不按章节配额 —— 配几张由模型看画面内容自己决定。
    """
    matches = list(PARA_RE.finditer(md))
    if not matches:
        print("[配图] 文档里没有段落时间标记，跳过配图", file=sys.stderr)
        return md

    keep = judge_shots(frames, work, title) if frames else set()
    assets = work / ASSETS_DIR
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir()

    # 帧归到"讲到这里"的那个段落：第一个结束时间 ≥ 帧时间的段落。
    # 同一段落只留最早的那张 —— 一个段落后堆好几张图会把正文冲散。
    ends = [_to_sec(m.group(2)) for m in matches]
    by_para: dict[int, list[str]] = {}
    for i in sorted(keep):
        f = frames[i]
        pi = next((k for k, t in enumerate(ends) if t >= f["t"]), len(ends) - 1)
        if pi in by_para:
            continue
        name = f"p{pi + 1:02d}-{f['path'].name}"
        shutil.copy(f["path"], assets / name)
        by_para[pi] = [name]

    # 从后往前替换，前面段落的偏移才不受影响
    out = md
    for pi in range(len(matches) - 1, -1, -1):
        m = matches[pi]
        tail = f"\n\n*{_time_link(ends[pi], src)}*"
        for name in by_para.get(pi, []):
            tail += f"\n\n![]({ASSETS_DIR}/{name})"
        out = out[:m.start()] + m.group(1) + tail + "\n" + out[m.end():]

    n = sum(len(v) for v in by_para.values())
    print(f"[配图] 保留 {n} 张，分布在 {len(by_para)}/{len(matches)} 个段落 → {assets}")
    return out


# ---------------------------------------------------------------- 自检

def _selftest():
    vtt = """WEBVTT
Kind: captions
Language: zh

00:00:01.000 --> 00:00:03.000
大家好

00:00:02.000 --> 00:00:04.000
大家好我们今天

00:00:03.000 --> 00:00:05.000
大家好我们今天讲神经网络

00:00:05.500 --> 00:00:08.000 align:start position:0%
<c>反向传播</c>算法
"""
    segs = parse_subs(vtt)
    assert len(segs) == 4, segs
    m = merge_segments(segs)
    # 滚动字幕被吸收成一条，且取最全的文本
    assert len(m) == 1, m
    assert m[0]["text"] == "大家好我们今天讲神经网络反向传播算法", m[0]["text"]
    assert m[0]["start"] == 1.0 and m[0]["end"] == 8.0, m[0]

    srt = """1
00:00:01,000 --> 00:00:02,000
Hello there

2
00:00:02,100 --> 00:00:04,000
world
"""
    m2 = merge_segments(parse_subs(srt))
    assert m2[0]["text"] == "Hello there world", m2[0]["text"]  # 英文补空格
    assert "00:00:01,000 --> 00:00:04,000" in to_srt(m2), to_srt(m2)
    assert _json_array('```json\n["a","b"]\n```') == ["a", "b"]
    assert _json_array("不是 JSON") is None
    assert slugify("MIT 6.824 分布式系统/Lecture 1") == "MIT-6-824-分布式系统-Lecture-1"
    assert _lang_rank(Path("source.zh-CN.vtt")) < _lang_rank(Path("source.en.vtt"))
    assert _lang_rank(Path("source.zh-Hans-ar.vtt")) == len(LANG_PREF)  # 认不出的排最后

    # 内容为纯数字的字幕不能被当成 SRT 序号吞掉（讲数字识别的课满地都是）
    srt_num = "1\n00:00:01,000 --> 00:00:02,000\n3\n\n2\n00:00:02,000 --> 00:00:03,000\n是一个数字\n"
    assert [s["text"] for s in parse_subs(srt_num)] == ["3", "是一个数字"]

    # 毫秒进位不能产出非法的 00:00:60,000
    assert _fmt_ts(59.9999) == "00:01:00,000", _fmt_ts(59.9999)
    assert _fmt_ts(3661.5) == "01:01:01,500", _fmt_ts(3661.5)
    assert _parse_ts("01:01:01,500") == 3661.5

    # 中文间的半角标点转全角，但不能误伤数字和英文
    assert fix_cjk_punct("交叉熵,那么") == "交叉熵，那么"
    assert fix_cjk_punct("结束.") == "结束。"
    assert fix_cjk_punct("模型1,000个参数") == "模型1,000个参数"
    assert fix_cjk_punct("精度3.14是") == "精度3.14是"

    # 段落时间标记的解析（配图对齐 + 时间链接都靠它）
    doc = "# 标题\n\n## 小节\n\n老师讲的第一段话。\n<!-- 01:30 -->\n\n第二段话。\n<!-- 05:45 -->\n"
    assert PARA_RE.findall(doc) == [("老师讲的第一段话。", "01:30"), ("第二段话。", "05:45")]
    assert _to_sec("05:45") == 345
    assert _mmss(90) == "01:30" and _mmss(3661) == "61:01"
    # 时间链接：有视频地址就做成可跳转的，没有就只显示时间
    assert _time_link(345, "https://youtu.be/abc") == "[05:45](https://youtu.be/abc?t=345)"
    assert _time_link(345, "https://x.com/v?a=1") == "[05:45](https://x.com/v?a=1&t=345)"
    assert _time_link(345, None) == "05:45"
    print("selftest ok")


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="课程视频 → 字幕 + 文案")
    ap.add_argument("src", nargs="?", help="本地视频文件或 URL")
    ap.add_argument("--from", dest="from_step", choices=STEPS, help="从该步重跑")
    ap.add_argument("--only", choices=["subs", "doc"], help="只跑到该步")
    ap.add_argument("--shots", action="store_true",
                    help="下载 720p 视频并给笔记配图（比只下音轨多约 130MB）")
    ap.add_argument("--list", action="store_true",
                    help="列出播放列表的分集和编号，不下载")
    ap.add_argument("--ep", type=int, metavar="N", help="只处理播放列表第 N 集")
    ap.add_argument("--work", default="work")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not a.src:
        ap.error("需要 src（或 --selftest）")

    src = a.src
    if a.list or a.ep:
        eps = playlist_entries(src)
        if not eps:
            sys.exit("取不到播放列表分集，确认链接里带 list= 参数")
        if a.list:
            for e in eps:
                print(f"{e['index']:>3}  {e['title']}")
            return
        hit = next((e for e in eps if e["index"] == a.ep), None)
        if not hit:
            sys.exit(f"没有第 {a.ep} 集（共 {len(eps)} 集，用 --list 看编号）")
        print(f"[选集] 第 {a.ep}/{len(eps)} 集：{hit['title']}")
        src = hit["url"]

    if src.startswith(("http://", "https://")):
        r = run([*YTDLP, "--print", "%(title)s", "--skip-download", src], check=False)
        title = r.stdout.strip() or src
    else:
        p = Path(src).expanduser()
        if not p.exists():
            sys.exit(f"文件不存在: {p}")
        title = p.stem

    work = Path(a.work).expanduser() / slugify(title)
    work.mkdir(parents=True, exist_ok=True)
    start = STEPS.index(a.from_step) if a.from_step else 0
    print(f"[课程] {title}\n[目录] {work}")

    video = fetch_video(src, work, want_video=a.shots)
    if a.shots and not has_video_stream(video):
        print("[配图] 源文件不含视频流，跳过配图", file=sys.stderr)
        a.shots = False

    def step(name):
        return STEPS.index(name) >= start

    subs_p = work / "subs.json"
    if step("subs") or not subs_p.exists():
        subs_p.write_text(json.dumps(build_subs(video, work), ensure_ascii=False, indent=1))
    segs = json.loads(subs_p.read_text())
    dur = segs[-1]["end"] if segs else 0
    print(f"[字幕] {len(segs)} 条 / {dur / 60:.1f} 分钟")

    if a.only == "subs":
        (work / "transcript.srt").write_text(to_srt(segs))
        print(f"[字幕] → {work / 'transcript.srt'}（未经 LLM 校对）")
        return

    clean_p = work / "clean.json"
    if step("clean") or not clean_p.exists():
        clean_p.write_text(json.dumps(clean_subs(segs, title), ensure_ascii=False, indent=1))
    clean = json.loads(clean_p.read_text())
    (work / "transcript.srt").write_text(to_srt(clean))
    print(f"[字幕] → {work / 'transcript.srt'}")

    doc_p = work / "course.md"
    # 配图靠章节时间标记对齐，所以 --shots 时文案得重生成一次
    if step("doc") or not doc_p.exists() or (a.shots and "](assets/" not in doc_p.read_text()):
        md = make_doc(clean, title, want_shots=a.shots)
        if a.shots:
            md = attach_shots(md, extract_frames(video, work), work, title,
                              src if src.startswith("http") else None)
        doc_p.write_text(md)
    print(f"[文案] → {doc_p}")


if __name__ == "__main__":
    main()
