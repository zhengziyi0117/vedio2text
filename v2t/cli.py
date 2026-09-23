"""命令行入口：解析参数、挑选集、按步骤跑主流程。"""
import argparse
import json
import sys
from pathlib import Path

from .doc import extract_frames, finish_doc, make_doc
from .llm import LLMConfigError, clean_subs, llm, selected_model, selected_provider
from .media import YTDLP, build_subs, fetch_video, has_video_stream, playlist_entries, run, slugify
from .selftest import run_selftest
from .subs import to_srt
from .workflow import DraftError, prepare_bundle, render_bundle

STEPS = ["subs", "clean", "doc"]


def _prepare(argv):
    ap = argparse.ArgumentParser(prog="v2t prepare", description="只准备 Work 写稿所需的素材")
    ap.add_argument("src", help="本地视频文件或 URL")
    ap.add_argument("--ep", type=int, metavar="N", help="播放列表第 N 集")
    ap.add_argument("--series", metavar="NAME", help="课程系列目录")
    ap.add_argument("--work", default="work", help="产物根目录")
    ap.add_argument("--shots", action="store_true", help="抽取候选画面供 Work 挑选")
    ap.add_argument("--input", choices=["subs", "clean"], default="subs",
                    help="给 Work 的字幕来源；clean 复用已有 clean.json，不调用模型")
    ap.add_argument("--refresh-subs", action="store_true", help="重新生成 subs.json")
    ap.add_argument("--chunk-seconds", type=int, default=600, help="单块最长秒数")
    ap.add_argument("--chunk-chars", type=int, default=9000, help="单块最大字符数")
    a = ap.parse_args(argv)
    if a.chunk_seconds <= 0 or a.chunk_chars <= 0:
        ap.error("分块时长和字符数必须大于 0")
    if a.refresh_subs and a.input == "clean":
        ap.error("--refresh-subs 与 --input clean 不能同时使用，避免复用过期校对结果")

    src = a.src
    if a.ep:
        eps = playlist_entries(src)
        hit = next((e for e in eps if e["index"] == a.ep), None)
        if not hit:
            ap.error(f"播放列表中没有第 {a.ep} 集")
        src = hit["url"]
    if src.startswith(("http://", "https://")):
        r = run([*YTDLP, "--print", "%(title)s", "--skip-download", src], check=False)
        title = r.stdout.strip() or src
    else:
        p = Path(src).expanduser()
        if not p.exists():
            ap.error(f"文件不存在：{p}")
        title = p.stem
    work = Path(a.work).expanduser()
    if a.series:
        work /= slugify(a.series)
    work /= slugify(title)
    work.mkdir(parents=True, exist_ok=True)

    # --input clean can reuse old clean.json without fetching media again.
    input_name = "clean.json" if a.input == "clean" else "subs.json"
    input_path = work / input_name
    if a.input == "clean" and not input_path.exists():
        ap.error(f"{input_path} 不存在；--input clean 只复用已有校对结果")
    need_video = (not input_path.exists() or a.refresh_subs or a.shots)
    video = fetch_video(src, work, want_video=a.shots) if need_video else None
    subs_path = work / "subs.json"
    if a.refresh_subs or (a.input == "subs" and not subs_path.exists()):
        subs_path.write_text(json.dumps(build_subs(video, work), ensure_ascii=False, indent=1),
                             encoding="utf-8")
    if not input_path.exists():
        ap.error(f"{input_path} 不存在；--input clean 只复用已有校对结果")
    if a.shots and not has_video_stream(video):
        print("[配图] 源文件不含视频流，跳过抽帧", file=sys.stderr)
        a.shots = False
    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        frames = extract_frames(video, work) if a.shots else []
        manifest = prepare_bundle(work, title, src, raw, frames, input_name,
                                  a.chunk_seconds, a.chunk_chars)
    except (DraftError, ValueError) as exc:
        ap.error(str(exc))
    print(f"[Work 素材] {manifest['segments']} 条字幕 / "
          f"{len(manifest['chunks'])} 块 / {len(manifest['frames'])} 张候选帧 → {work}")
    print(f"[下一步] 填写 {work / 'draft' / 'metadata.json'} 和各块草稿，再运行 "
          f"uv run -m v2t render '{work}' --check")


def _render(argv):
    ap = argparse.ArgumentParser(prog="v2t render", description="校验 Work 草稿并生成 course.md")
    ap.add_argument("episode_dir", type=Path, help="prepare 输出的单讲目录")
    ap.add_argument("--check", action="store_true", help="只校验，不写入文件")
    ap.add_argument("--force", action="store_true", help="允许覆盖已有 course.md")
    a = ap.parse_args(argv)
    try:
        result = render_bundle(a.episode_dir.expanduser(), a.force, a.check)
    except (DraftError, OSError, ValueError, KeyError, TypeError) as exc:
        ap.error(str(exc))
    print(f"[Work 稿件] {result['chunks']} 块 / {result['segments']} 条字幕 / "
          f"{result['paragraphs']} 段 / {result['images']} 张图："
          f"{'校验通过' if a.check else a.episode_dir / 'course.md'}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        return _prepare(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        return _render(sys.argv[2:])
    ap = argparse.ArgumentParser(description="课程视频 → 字幕 + 文案")
    ap.add_argument("src", nargs="?", help="本地视频文件或 URL")
    ap.add_argument("--from", dest="from_step", choices=STEPS, help="从该步重跑")
    ap.add_argument("--only", choices=["subs", "doc"], help="只跑到该步")
    ap.add_argument("--shots", action="store_true",
                    help="下载 720p 视频并给笔记配图（比只下音轨多约 130MB）")
    ap.add_argument("--list", action="store_true",
                    help="列出播放列表的分集和编号，不下载")
    ap.add_argument("--ep", type=int, metavar="N", help="只处理播放列表第 N 集")
    ap.add_argument("--series", metavar="NAME",
                    help="归到某门课下面（work/<NAME>/<讲名>/），出书时这门课自成一组")
    ap.add_argument("--work", default="work")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--llm-test", action="store_true",
                    help="用当前配置发送一次最小模型请求，检查后端是否真的可用")
    a = ap.parse_args()

    if a.selftest:
        return run_selftest()
    if a.llm_test:
        try:
            provider = selected_provider()
        except LLMConfigError as e:
            ap.error(str(e))
        model = selected_model(provider) or "Codex 本地配置"
        print(f"[LLM] 后端={provider} 模型={model}")
        print(llm("只返回字符串 OK，不要解释。",
                  "请严格回复 OK。", max_tokens=16, think=False))
        return
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

    work = Path(a.work).expanduser()
    if a.series:
        work /= slugify(a.series)
    work /= slugify(title)
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
    # 先不带 --shots 跑过、后来才加 --shots 的，得重生成一次才有配图
    if step("doc") or not doc_p.exists() or (a.shots and "](assets/" not in doc_p.read_text()):
        md = make_doc(clean, title)
        # 时间链接每篇都做；抽帧只有 --shots 才值得（多下 720p 视频）
        md = finish_doc(md, extract_frames(video, work) if a.shots else [],
                        work, title, src if src.startswith("http") else None, a.shots)
        doc_p.write_text(md)
    print(f"[文案] → {doc_p}")
