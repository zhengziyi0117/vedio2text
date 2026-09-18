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
    if a and b and a[-1].isascii() and b[0].isascii() and b[0].isalnum() and \
       (a[-1].isalnum() or a[-1] in ",.;:!?)]}'\""):
        return a + " " + b
    return a + b


def _novel_after_overlap(previous: str, current: str) -> str | None:
    """返回滚动字幕 ``current`` 中没有出现在 ``previous`` 里的尾部。

    YouTube 的英文自动字幕通常不是简单的连续片段，而是一个不断向右
    滚动的窗口。中文可以用“新 cue 是旧 cue 的超集”处理，英文则常见为
    ``previous`` 的后半段与 ``current`` 的前半段重叠，因此需要按词边界
    找最长后缀/前缀匹配。返回 ``None`` 表示两条 cue 没有可靠重叠。
    """
    previous = previous.strip()
    current = current.strip()
    if not previous or not current:
        return current
    if current.startswith(previous):
        return current[len(previous):].lstrip()
    if previous.endswith(current):
        return ""

    old_words = previous.split()
    new_words = current.split()
    for n in range(min(len(old_words), len(new_words)), 0, -1):
        overlap = " ".join(new_words[:n])
        if previous.endswith(overlap) and current.startswith(overlap):
            return current[len(overlap):].lstrip()

    # 中文和没有空格的字幕按字符边界处理；至少两个字符才认为是重叠，
    # 避免把普通英文单字母/标点巧合当成滚动字幕。
    for n in range(min(len(previous), len(current)), 1, -1):
        if previous.endswith(current[:n]):
            return current[n:].lstrip()
    return None


def _last_sentence_boundary(text: str) -> int | None:
    """返回最后一个适合切段的句末位置（不含后续空白）。"""
    hits = [m.end() for m in re.finditer(r"[.!?。！？](?:[\"'”’）)]*)", text)]
    return hits[-1] if hits else None


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
    current = None
    raw_previous = ""

    def flush():
        nonlocal current
        if current and current["text"].strip():
            current["text"] = current["text"].strip()
            out.append(current)
        current = None

    def append_new_text(text, start, end):
        """把去重后的新文本接到当前段，必要时按句末切开。"""
        nonlocal current
        start = max(start, out[-1]["end"] if out else start)
        if not text:
            if current:
                current["end"] = max(current["end"], end)
            return
        if current is None:
            current = {"start": start, "end": end, "text": text}
            return

        # 滚动 cue 在上一句结束后才带出下一句；先落盘上一句，避免把
        # “inscrutable. Um ...” 拼成一个跨句的超长段落。
        if _last_sentence_boundary(current["text"]) == len(current["text"].rstrip()):
            flush()
            current = {"start": max(start, out[-1]["end"]), "end": end, "text": text}
            return

        current["text"] = _join(current["text"], text)
        current["end"] = max(current["end"], end)

        boundary = _last_sentence_boundary(current["text"])
        if boundary and boundary < len(current["text"].rstrip()):
            head = current["text"][:boundary].strip()
            tail = current["text"][boundary:].strip()
            current["text"] = head
            flush()
            current = {"start": max(start, out[-1]["end"]), "end": end, "text": tail}
        # 滚动字幕的窗口本身可能已经超过 max_len；只要还没有到时间上限，
        # 继续等句末可以避免把一个长句切在“前半个窗口”上。
        elif current["end"] - current["start"] > max_dur:
            flush()

    for s in segs:
        t = (s.get("text") or "").strip()
        if not t:
            continue
        if current is None:
            current = {"start": s["start"], "end": s["end"], "text": t}
            raw_previous = t
            continue

        # 先针对相邻原始 cue 去重。这个判断必须在 max_len 约束之前，
        # 否则英文长句一旦超过 60 字符，就会把重复窗口原样保留下来。
        novel = _novel_after_overlap(raw_previous, t) if raw_previous else None
        if novel is not None:
            append_new_text(novel, s["start"], s["end"])
            raw_previous = t
            continue

        # 非滚动字幕仍按原来的时间间隔和长度规则合并。
        if (s["start"] - current["end"] < max_gap or t == current["text"]) and \
           len(current["text"]) + len(t) <= max_len and \
           s["end"] - current["start"] <= max_dur:
            append_new_text(t, s["start"], s["end"])
        else:
            flush()
            start = max(s["start"], out[-1]["end"] if out else s["start"])
            current = {"start": start, "end": s["end"], "text": t}
        raw_previous = t
    flush()
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
