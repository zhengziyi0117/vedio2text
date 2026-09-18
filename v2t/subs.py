"""字幕文本层：SRT/VTT 解析、句子级归并、时间戳、中文标点规整。

只跟文本打交道，不碰网络和模型。
"""
import re


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


_HALF_PUNCT = {
    ",": "，", ";": "；", ":": "：", "?": "？", "!": "！", ".": "。",
}
# 只动"中文…中文"之间的半角标点，避免误伤 1,000 / 3.14 / e.g.
_CJK_BETWEEN = re.compile(r"(?<=[一-鿿])([,;:?!.])(?=[一-鿿]|$)")


def fix_cjk_punct(s: str) -> str:
    return _CJK_BETWEEN.sub(lambda m: _HALF_PUNCT[m.group(1)], s)
