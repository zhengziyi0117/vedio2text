"""纯函数自检：解析、时间戳、标点、段落标记、模型输出的清洗。不联网。"""
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from .doc import (PARA_RE, _mad, _mmss, _settled_index, _shot_pick, _time_link,
                  _to_sec, finish_doc, thin_cuts)
from .llm import _openai_input, _sdk_prompt, json_array
from .media import LANG_PREF, _lang_rank, slugify
from .subs import _fmt_ts, _parse_ts, fix_cjk_punct, merge_segments, parse_subs, to_srt
from .workflow import DraftError, prepare_bundle, render_bundle


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
    blocks = [
        {"type": "text", "text": "挑图"},
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": "aGk="}},
    ]
    openai_input = _openai_input(blocks)
    assert openai_input == [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "挑图"},
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,aGk=",
                "detail": "high",
            },
        ],
    }], openai_input
    assert _openai_input("纯文本") == [{
        "role": "user",
        "content": [{"type": "input_text", "text": "纯文本"}],
    }]
    assert openai_input[0]["content"][1]["image_url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert slugify("MIT 6.824 分布式系统/Lecture 1") == "MIT-6-824-分布式系统-Lecture-1"
    # 长标题不能截到分不出讲次（CS336 那串标题前 60 字符全都一样）
    t = "Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture %d: x"
    assert slugify(t % 1) != slugify(t % 2), slugify(t % 1)
    # 默认英文优先：非英文课挂的 zh.* 是 YouTube 机翻，拿机翻当原文等于白劣化一遍，
    # 中文由文稿那步翻译。中文课靠 V2T_LANG=zh 顶回去。
    assert _lang_rank(Path("source.en.vtt")) < _lang_rank(Path("source.zh-Hans.vtt"))
    assert _lang_rank(Path("source.en-US.vtt")) < _lang_rank(Path("source.zh-CN.vtt"))
    # en-orig 是 YouTube 标的原始音轨，得和 en 同级，别输给机翻中文轨
    assert _lang_rank(Path("source.en-orig.vtt")) < _lang_rank(Path("source.zh-Hans.vtt"))
    assert _lang_rank(Path("source.zh-Hans-ar.vtt")) == len(LANG_PREF)  # 认不出的排最后
    # 从别的语言翻过来的中文轨也不是中文原文，同样排最后
    assert _lang_rank(Path("source.zh-Hans-en-US.vtt")) == len(LANG_PREF)
    # B 站的机翻码 ai-zh 得认成中文，不然设了 V2T_LANG 就被丢掉
    assert _lang_rank(Path("source.ai-zh.vtt")) < _lang_rank(Path("source.ab.vtt"))
    # yt-dlp 的英文轨叫 en-US，得认成英文；否则它跟 ab 并列排最后，
    # 按文件名排序时 source.ab.vtt 反而赢（第 11、14 讲踩过）
    assert _lang_rank(Path("source.en-US.vtt")) < _lang_rank(Path("source.ab.vtt"))
    assert _lang_rank(Path("source.en-US.vtt")) == _lang_rank(Path("source.en.vtt"))

    # 内容为纯数字的字幕不能被当成 SRT 序号吞掉（讲数字识别的课满地都是）
    srt_num = "1\n00:00:01,000 --> 00:00:02,000\n3\n\n2\n00:00:02,000 --> 00:00:03,000\n是一个数字\n"
    assert [s["text"] for s in parse_subs(srt_num)] == ["3", "是一个数字"]

    # 毫秒进位不能产出非法的 00:00:60,000
    assert _fmt_ts(59.9999) == "00:01:00,000", _fmt_ts(59.9999)
    assert _fmt_ts(3661.5) == "01:01:01,500", _fmt_ts(3661.5)
    assert _parse_ts("01:01:01,500") == 3661.5
    # WebVTT 省掉小时位（ffmpeg 转出来的字幕：一小时内是 MM:SS.mmm，
    # 之后才带小时）。只认 H:MM:SS 会把前半段 cue 全丢光。
    short_vtt = """WEBVTT

00:00.920 --> 00:03.000
前半段

01:00:02.590 --> 01:00:11.729
后半段
"""
    assert parse_subs(short_vtt) == [
        {"start": 0.92, "end": 3.0, "text": "前半段"},
        {"start": 3602.59, "end": 3611.729, "text": "后半段"},
    ], parse_subs(short_vtt)

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

    # 密集检测只留每簇最早那个，判据是**跟上一个检测点**比：
    # 一次淡入会连着好几帧越过阈值，机位每 3 秒重取景也是一串，都得收成一簇。
    assert thin_cuts([1.0, 1.2, 1.5, 12.0, 12.2, 12.4, 30.0], gap=4) == [1.0, 12.0, 30.0]
    assert thin_cuts([], gap=4) == []
    assert thin_cuts([5.0], gap=4) == [5.0]
    # 跟"上一个保留点"比会退化成每 4 个留一个：3 秒一串的机位周期会凑出精确
    # 12.00 秒间隔活下来（实测就有 58 个），整串都留 = 没瘦身
    assert thin_cuts([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0], gap=4) == [0.0]
    # 被丢掉的那个会顶替基准：9.0 之后 2 秒的 11.0 算同一簇、只留 9.0。
    # 好处是机位那种 3 秒等距串整串收成一帧，代价是紧跟在一串之后的真切换会被吞。
    assert thin_cuts([0.0, 9.0, 11.0, 25.0], gap=4) == [0.0, 9.0, 25.0]

    # 抽帧挑"画面已经静止"的那一帧。窗口末尾是淡入中段（帧间差大），往前扫到
    # 第一对没动的帧为止 —— 实测 Lecture-4 的 648.83 那个窗口就是这个形状
    assert _settled_index([0.0, 0.0, 0.006, 0.194, 1.59, 9.4]) == 3
    assert _settled_index([9.4, 4.5, 1.2, 0.8]) is None   # 整段都在动（机位跟拍）→ 退回固定偏移
    assert _settled_index([]) is None
    assert _settled_index([0.0]) == 1                     # 只有两帧且没动
    assert _mad(b"\x00\x10", b"\x00\x20") == 8.0

    # 兜底走 Agent SDK：纯文本直接透传，图文块得包成流式输入的信封才收
    assert _sdk_prompt("hi") == "hi"

    async def _drain(g):
        return [m async for m in g]

    blocks = [{"type": "text", "text": "挑图"}]
    env = asyncio.run(_drain(_sdk_prompt(blocks)))
    assert env == [{"type": "user", "message": {"role": "user", "content": blocks}}], env

    # Work 接力：分块素材 → 分块草稿 → 校验 → 成稿，不能漏字幕或复用旧草稿。
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "frames").mkdir()
        frame = work / "frames" / "f0001.jpg"
        frame.write_bytes(b"jpeg fixture")
        raw = [
            {"start": 0, "end": 2, "text": "开头"},
            {"start": 2, "end": 4, "text": "第一节"},
            {"start": 12, "end": 14, "text": "第二节"},
            {"start": 14, "end": 16, "text": "结束"},
        ]
        manifest = prepare_bundle(work, "测试课程", "https://youtu.be/test", raw,
                                  [{"t": 3, "path": frame}], max_seconds=8)
        assert len(manifest["chunks"]) == 2, manifest
        meta = work / "draft" / "metadata.json"
        meta.write_text(json.dumps({"title": "测试课程", "intro": "这是导语。"}))
        for i, entry in enumerate(manifest["chunks"]):
            path = work / entry["draft"]
            draft = json.loads(path.read_text())
            draft["paragraphs"] = [{"start_id": entry["first_id"],
                                    "end_id": entry["last_id"],
                                    "text": f"这是第 {i + 1} 节完整内容。",
                                    **({"heading": "第一节", "frame_id": "f0001"}
                                       if i == 0 else {})}]
            path.write_text(json.dumps(draft, ensure_ascii=False))
        assert render_bundle(work, check=True) == {
            "chunks": 2, "segments": 4, "paragraphs": 2, "images": 1}
        assert not (work / "course.md").exists()  # --check 无副作用
        render_bundle(work)
        course = (work / "course.md").read_text()
        assert "*[00:04](https://youtu.be/test?t=4)*" in course, course
        assert "![](assets/work-p001-f0001.jpg)" in course, course
        try:
            render_bundle(work)
            assert False, "不应该覆盖现有成稿"
        except DraftError:
            pass
        draft_path = work / manifest["chunks"][1]["draft"]
        draft = json.loads(draft_path.read_text())
        draft["paragraphs"][0]["start_id"] = "s000004"  # 漏掉 s000003
        draft_path.write_text(json.dumps(draft))
        try:
            render_bundle(work, check=True)
            assert False, "漏字幕不能通过校验"
        except DraftError:
            pass
        draft["paragraphs"][0]["start_id"] = "s000003"
        draft_path.write_text(json.dumps(draft))
        raw[0]["text"] = "改过的素材"
        prepare_bundle(work, "测试课程", "https://youtu.be/test", raw,
                       [{"t": 3, "path": frame}], max_seconds=8)
        try:
            render_bundle(work, check=True)
            assert False, "素材变了后不能继续用旧草稿"
        except DraftError:
            pass

    # URL 的媒体被拒绝时，prepare 仍能用 yt-dlp 单独取回的字幕继续；
    # 没有媒体也没有字幕则必须明确失败，不能产出空课程。
    from . import cli
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = "https://youtu.be/test"
        episode = root / "测试视频"
        raw = [{"start": i * 2, "end": i * 2 + 1,
                "text": f"第 {i} 条字幕。"} for i in range(12)]

        def write_captions(_src, work):
            work.mkdir(parents=True, exist_ok=True)
            (work / "source.zh.srt").write_text(to_srt(raw), encoding="utf-8")

        common = (mock.patch.object(cli, "run", return_value=SimpleNamespace(stdout="测试视频\n")),
                  mock.patch.object(cli, "fetch_video", side_effect=SystemExit("媒体不可用")),
                  mock.patch.object(cli, "FORCE_ASR", False))
        with common[0], common[1], common[2], \
                mock.patch.object(cli, "download_subtitles", side_effect=write_captions):
            cli._prepare([source, "--work", str(root), "--shots"])
        assert (episode / "manifest.json").is_file()
        assert json.loads((episode / "manifest.json").read_text())["frames"] == []
        assert len(json.loads((episode / "subs.json").read_text())) > 0

        empty_root = root / "empty"
        with common[0], common[1], common[2], \
                mock.patch.object(cli, "download_subtitles"):
            try:
                cli._prepare([source, "--work", str(empty_root)])
                assert False, "无媒体与字幕时不能继续"
            except SystemExit:
                pass
        assert not (empty_root / "测试视频" / "manifest.json").exists()

    print("selftest ok")
