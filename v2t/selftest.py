"""纯函数自检：解析、时间戳、标点、段落标记、模型输出的清洗。不联网。"""
import asyncio
from pathlib import Path

from .doc import PARA_RE, _mmss, _shot_pick, _time_link, _to_sec, finish_doc
from .llm import _sdk_prompt, json_array
from .media import LANG_PREF, _lang_rank, slugify
from .subs import _fmt_ts, _parse_ts, fix_cjk_punct, merge_segments, parse_subs, to_srt


def run_selftest():
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

    # YouTube 英文自动字幕常把上一条 cue 的尾部滚动到下一条开头，
    # 不能因为英文长句超过 max_len 就把重复窗口保留下来。
    rolling_en = """WEBVTT

00:00:01.000 --> 00:00:02.000
So today we talk about

00:00:02.000 --> 00:00:03.000
So today we talk about architecture,

00:00:03.000 --> 00:00:04.000
architecture, which is difficult.

00:00:04.000 --> 00:00:05.000
which is difficult. Now we continue.
"""
    rolling = merge_segments(parse_subs(rolling_en))
    assert [s["text"] for s in rolling] == [
        "So today we talk about architecture, which is difficult.",
        "Now we continue.",
    ], rolling
    assert all(a["end"] <= b["start"] for a, b in zip(rolling, rolling[1:])), rolling

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
    assert json_array('```json\n["a","b"]\n```') == ["a", "b"]
    assert json_array("不是 JSON") is None
    assert slugify("MIT 6.824 分布式系统/Lecture 1") == "MIT-6-824-分布式系统-Lecture-1"
    # 长标题不能截到分不出讲次（CS336 那串标题前 60 字符全都一样）
    t = "Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture %d: x"
    assert slugify(t % 1) != slugify(t % 2), slugify(t % 1)
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
    # 标记甩在段尾同一行也得认（模型两种写法随机出）
    assert PARA_RE.findall("第一段。 <!-- 01:30 -->\n\n第二段。\n<!-- 05:45 -->\n") == [
        ("第一段。", "01:30"), ("第二段。", "05:45")]
    assert _to_sec("05:45") == 345
    assert _mmss(90) == "01:30" and _mmss(3661) == "61:01"
    # 时间链接：有视频地址就做成可跳转的，没有就只显示时间
    assert _time_link(345, "https://youtu.be/abc") == "[05:45](https://youtu.be/abc?t=345)"
    assert _time_link(345, "https://x.com/v?a=1") == "[05:45](https://x.com/v?a=1&t=345)"
    assert _time_link(345, None) == "05:45"

    # 不配图也每段带时间：frames 为空时只渲染链接，不碰 assets
    doc2 = finish_doc(doc, [], Path("/tmp"), "t", "https://youtu.be/x", want_shots=False)
    assert "*[01:30](https://youtu.be/x?t=90)*" in doc2, doc2
    assert "*[05:45](https://youtu.be/x?t=345)*" in doc2, doc2
    assert "![](" not in doc2, doc2
    # 本地文件没地址可跳，退化成纯时间
    assert "*05:45*" in finish_doc(doc, [], Path("/tmp"), "t", None, want_shots=False)

    # 挑图的回项是模型输出，越界/缺字段/非数字必须丢掉，不能脏到主流程
    assert _shot_pick({"n": 3}, 9) == 2
    assert _shot_pick({"n": 10}, 9) is None   # 超出本批编号
    assert _shot_pick({"n": 0}, 9) is None    # 编号从 1 起
    assert _shot_pick({}, 9) is None          # 缺 n
    assert _shot_pick({"n": "x"}, 9) is None
    assert _shot_pick("3", 9) is None

    # 兜底走 Agent SDK：纯文本直接透传，图文块得包成流式输入的信封才收
    assert _sdk_prompt("hi") == "hi"

    async def _drain(g):
        return [m async for m in g]

    blocks = [{"type": "text", "text": "挑图"}]
    env = asyncio.run(_drain(_sdk_prompt(blocks)))
    assert env == [{"type": "user", "message": {"role": "user", "content": blocks}}], env

    print("selftest ok")
