"""文稿：字幕 → markdown 讲稿，每段回填可跳回视频的时间链接，再按需挂配图。"""
import base64
import os
import re
import shutil
import sys
from pathlib import Path

from .llm import json_array, llm
from .media import run
from .subs import fix_cjk_punct

DOC_LANG = os.getenv("V2T_DOC_LANG", "中文")  # 文稿写成什么语言
ASSETS_DIR = "assets"   # 配图目录（相对 course.md）
SCENE_THRESHOLD = 0.1   # 场景切换灵敏度（白底幻灯片之间差异小，压低了才不漏）
SETTLE_BEFORE = 0.5     # 取切换点前多少秒的帧（切换点那帧多半在过渡动画中间）
SHEET = 9               # 每张拼图放几帧（3x3）
SHOT_BATCH = SHEET * 3  # 每批送审的帧数


DOC_SYS = f"""你在把一门课的完整字幕整理成一篇能替代看视频的文章。

核心要求：**忠实转写，不是提炼**。老师讲了什么就写什么 —— 不归纳、不压缩、
不补充自己的总结或评论。读者要的是"老师讲过的话"，不是你的读书笔记。

- 按老师讲述的顺序组织。**按意思分段**：讲完一个话题、一个例子、一段演示就另起一段，
  一段大约 3~6 句（150~400 字）。别写成上千字的一整块，也别拆成零散小段或要点清单
- 补上必要的衔接，让段落读起来连贯
- 保留老师的例子、比喻、演示步骤、强调语气和第一人称（"我们来看"、"我演示一下"）
- 保留所有实质内容。宁可长，也不要为了简洁丢信息
- 只去掉真正的口水词和口吃重复（嗯、啊、那个）
- 源字幕可能是没标点的机翻／ASR（B 站 ai-zh 那种），带叠字重复（"再再再再"）、
  同音错字、一逗到底。这些由你在文稿里消化掉：补标点断句、合并重复，写成通顺的
  书面段落。字幕原文保留在 subs.json / transcript.srt 里，不必照搬进文稿
- 第一行用 `#` 写一句话标题（出书的章节目录取这行），第二行再用一句话说明这堂课讲什么
- 用 `##` 分小节，标题写具体，别用"第一部分"这种
- 听不清或不确定的内容不要编，宁可不写
- 文稿用{DOC_LANG}写；源字幕是别的语言就翻译过来，专业术语首次出现时括注原文"""

# 每段末尾回填时间标记：渲染成能跳回视频的时间链接，也是配图的对齐锚点
DOC_TIME_HINT = """

输入的字幕每行带 [MM:SS] 时间标记。在此之上额外要求：
- 在**每一个自然段的末尾**附上该段对应的时间标记 `<!-- MM:SS -->`，
  取输入里出现过的 [MM:SS] 值（该段内容结束处那个），不要自己编。
- 标记独占一行，紧跟在段末，格式严格照抄，不要加别的字。"""


def _mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def make_doc(segs, title: str) -> str:
    """每行字幕前面挂 [MM:SS]，并要求模型逐段回填 <!-- MM:SS -->。"""
    body = "\n".join(f"[{_mmss(s['start'])}] {s['text']}" for s in segs)
    print(f"[文案] 全文 {len(body)} 字，送模型…")
    md = llm(DOC_SYS + DOC_TIME_HINT,
             f"课程标题：{title}\n\n完整字幕：\n{body}", 48000)
    return fix_cjk_punct(md.strip())


# 自然段 + 紧随其后的时间标记（由 DOC_TIME_HINT 让模型产出）。
# 它既是配图的对齐锚点，最终也会渲染成能跳回视频的时间链接。
# 不用 ^ 锚定行首：段落可以是正文也可以是列表项，只要下一行是标记就算。
# 模型一半时候把标记甩在段尾同一行，所以两者都收。
PARA_RE = re.compile(r"([^\n]+?)\s*<!--\s*(\d{1,2}:\d{2})\s*-->")


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
    """在场景切换处抽帧。返回 [{t, path}] 按时间升序。

    切换点本身是淡入/翻页的中间态（半透明、两页叠着），干净的画面在切换**前**，
    所以取切换点前 SETTLE 秒那一帧。阈值压得低：这套白底幻灯片的切换，
    ffmpeg 给的分往往到不了 0.4，漏掉的比误报的多得多。
    """
    out = work / "frames"
    if (work / "frames.txt").exists():
        times = [float(t) for t in (work / "frames.txt").read_text().split()]
        return [{"t": t, "path": p} for t, p in zip(times, sorted(out.glob("*.jpg")))]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[配图] 场景检测抽帧（阈值 {threshold}）…")
    r = run(["ffmpeg", "-v", "info", "-i", str(video), "-an",
             "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"], check=False)
    cuts = [float(t) for t in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    # 逐个 -ss 取帧：webm 快进够快，比再整段解码一遍便宜
    times = []
    for t in cuts:
        at = max(0.0, t - SETTLE_BEFORE)
        dst = out / f"f{len(times) + 1:04d}.jpg"
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "3", str(dst)], check=False)
        if dst.exists():
            times.append(at)
    (work / "frames.txt").write_text("\n".join(f"{t:.2f}" for t in times))
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
- 整帧只有讲师、纯文字封面、过渡动画、空白或模糊。注意课堂录像常是双拼画面
  （一边讲师、一边幻灯片/终端/黑板），这种按有内容的那半边判断，
  别因为有讲师在里面就整帧丢掉
- 只是把正文原话打在屏幕上的字幕式画面，正文自己讲清楚了就不需要图
- 和已选画面重复的（同一张图的不同时刻只留最完整的一张）

宁可少配也不要凑数，但别漏：以幻灯片、板书或终端演示为主的课，
一堂 100 分钟通常能找到 20 张以上值得配的画面；只挑出个位数说明标准收得太紧了。

用户会一并给你这篇文章的各段落（编号、结束时间、正文），以及每个候选帧的时间。
两边都用得上：

- **拿段落正文排除"已经讲清楚了"的画面** —— 有些截图只是把正文原话打在屏幕上，
  那段正文读一遍就够了，不用再配图。
- **一段最多配一张**。一段的时间范围是「上一段结束时间，本段结束时间」，
  落进同一个范围里的候选属于同一段。同段有多个候选时只挑信息最完整的那个 ——
  同一张幻灯片的不同时刻、讲者出镜、过渡动画，都不算"另一个候选"。

输出 JSON 数组，每项 {"n": 格子编号}，只列值得配的，不配的不用出现在数组里。
图挂到哪一段由程序按时间戳决定，不用你管。只输出 JSON 数组，不要代码块，不要解释。"""


def _img_block(p: Path) -> dict:
    data = base64.standard_b64encode(p.read_bytes()).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def _para_list(paras: list[dict]) -> str:
    """段落清单：模型靠它判断正文是不是已经把这画面讲清楚了。"""
    return "文章段落：\n\n" + "\n\n".join(
        f"{i}. （{p['time']}）{p['text']}" for i, p in enumerate(paras, 1))


def _shot_pick(item, n_frames: int):
    """模型回的一项 {"n": 格子号} → 帧下标。

    模型输出是外部输入，越界、缺字段、非数字一律丢，别让它脏到主流程。
    """
    if not isinstance(item, dict):
        return None
    try:
        n = int(item["n"]) - 1
    except (KeyError, TypeError, ValueError):
        return None
    return n if 0 <= n < n_frames else None


def judge_shots(cand: list[dict], work: Path, title: str,
                paras: list[dict]) -> set[int]:
    """分批送审（一次看太多帧模型会乱），返回值得配图的 cand 下标集合。

    带文章段落一起看，模型才判断得了"这段正文是不是已经把画面讲清楚了"。
    每帧的时间也一并给它 —— 它得知道哪些候选属于同一段，才能执行"一段一张"。
    挂到哪一段不归它管 —— 那是时间戳的事，见 finish_doc。
    """
    intro = _para_list(paras)
    keep: set[int] = set()
    for base in range(0, len(cand), SHOT_BATCH):
        chunk = cand[base:base + SHOT_BATCH]
        stamps = ", ".join(f"{n}={_mmss(f['t'])}" for n, f in enumerate(chunk, 1))
        content = [{"type": "text", "text": intro},
                   *[_img_block(s) for s in make_sheets(chunk, work)],
                   {"type": "text",
                    "text": f"课程：{title}\n本批 {len(chunk)} 个候选。"
                            f"编号=时间：{stamps}"}]
        # 跟 _clean_block 一样重试一次：这批 27 帧，偶发一次失败就整批不判，
        # 白丢一段的配图。注意 [] 是合法的"这批没有值得配的"（比如一整批都是
        # 同一段动画的帧），不是失败 —— 只有解析不出来（None）才重试。
        arr = None
        for _ in range(2):
            arr = json_array(llm(SHOT_SYS, content, 8000, think=False))
            if arr is not None:
                break
        if arr is None:
            print(f"  ! 第 {base // SHOT_BATCH + 1} 批挑图失败，跳过", file=sys.stderr)
            continue
        for item in arr:
            i = _shot_pick(item, len(chunk))
            if i is not None:
                keep.add(base + i)
    return keep


def finish_doc(md: str, frames: list[dict], work: Path, title: str,
               src: str | None, want_shots: bool = True) -> str:
    """收尾：每段渲染可跳回视频的时间链接；want_shots 时再挂截图。

    时间链接跟配图是两回事，所以每篇文稿都做；frames 为空或
    want_shots=False 就只做链接。配几张由模型带着正文判断（哪些画面正文已经
    讲清楚了、不用配），挂到哪一段由代码按段落时间范围定，保证图和它自己的
    时间链接指向同一段时间。
    """
    matches = list(PARA_RE.finditer(md))
    if not matches:
        print("[时间] 文稿里没有段落时间标记 —— 模型没按格式回填，"
              "这篇既没有时间链接也不会有配图", file=sys.stderr)
        return md

    ends = [_to_sec(m.group(2)) for m in matches]
    paras = [{"time": m.group(2), "text": m.group(1)} for m in matches]
    keep = judge_shots(frames, work, title, paras) if (want_shots and frames) else set()

    assets = work / ASSETS_DIR
    if keep:
        shutil.rmtree(assets, ignore_errors=True)
        assets.mkdir()

    # 帧只能挂到"正在讲它"的那一段：第一个结束时间 ≥ 帧时间的段落。
    # 段落结束时间单调递增，把时间轴切成不重叠的区间，所以每帧的归属唯一 ——
    # 图必然来自它旁边那个 *[MM:SS]* 链接指向的时间段，读者点进去看到的就是它。
    # 一段最多一张：同段有多个候选时挑最靠近段末的（幻灯片放到最后最完整）。
    best: dict[int, int] = {}
    for i in sorted(keep):
        pi = next((k for k, t in enumerate(ends) if t >= frames[i]["t"]), None)
        if pi is None:
            continue          # 末段结束之后的帧没有归属，丢掉
        cur = best.get(pi)
        if cur is None or abs(frames[i]["t"] - ends[pi]) < abs(frames[cur]["t"] - ends[pi]):
            best[pi] = i

    by_para: dict[int, list[str]] = {}
    for pi, i in sorted(best.items()):
        name = f"p{pi + 1:02d}-{frames[i]['path'].name}"
        shutil.copy(frames[i]["path"], assets / name)
        by_para[pi] = [name]

    # 从后往前替换，前面段落的偏移才不受影响
    out = md
    for pi in range(len(matches) - 1, -1, -1):
        m = matches[pi]
        tail = f"\n\n*{_time_link(ends[pi], src)}*"
        for name in by_para.get(pi, []):
            tail += f"\n\n![]({ASSETS_DIR}/{name})"
        out = out[:m.start()] + m.group(1) + tail + "\n" + out[m.end():]

    print(f"[时间] {len(matches)} 个段落都带上了时间链接"
          + ("（源是本地文件，只显示时间不可跳转）" if not (src or "").startswith("http") else ""))
    if keep:
        # 一段最多一张，所以落盘数看 by_para 而不是 keep（keep 是去重前的候选数）
        print(f"[配图] 模型选中 {len(keep)} 个候选，一段一张后挂上 {len(by_para)} 张"
              f"（共 {len(matches)} 段）→ {assets}")
    return out
