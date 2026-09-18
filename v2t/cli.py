"""命令行入口：解析参数、挑选集、按步骤跑主流程。"""
import argparse
import json
import sys
from pathlib import Path

from .doc import extract_frames, finish_doc, make_doc
from .llm import clean_subs
from .media import YTDLP, build_subs, fetch_video, has_video_stream, playlist_entries, run, slugify
from .selftest import run_selftest
from .subs import to_srt

STEPS = ["subs", "clean", "doc"]


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
    ap.add_argument("--series", metavar="NAME",
                    help="归到某门课下面（work/<NAME>/<讲名>/），出书时这门课自成一组")
    ap.add_argument("--work", default="work")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return run_selftest()
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
