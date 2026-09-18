"""调模型：有 key 直连 Anthropic API，没有就回退 Claude Agent SDK。

外加字幕校对 —— 术语表和分批校对都是机械 JSON 任务，`think=False` 关思考链。
"""
import asyncio
import concurrent.futures as cf
import json
import os
import re
import sys

from .subs import fix_cjk_punct

# 优先跟随 Claude Code 已配好的 ANTHROPIC_MODEL（走自建网关时通常就是这个），
# 免得脚本自己写死的模型名把用户的配置顶掉、还多花钱
MODEL = os.getenv("V2T_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-opus-5"
CHUNK = 40          # 每块送 LLM 的字幕条数
WORKERS = 4


def llm(system: str, user, max_tokens: int = 16000, think: bool = True) -> str:
    """user 传字符串，或 content blocks 列表（图文混排）。

    有 API key 或 auth token 就直连（ANTHROPIC_BASE_URL 会一并生效，
    所以走自建网关、只配 ANTHROPIC_AUTH_TOKEN 的情况也能用）；
    两者都没有才回退到 Claude Agent SDK（底层就是本机 Claude Code）。

    think=False 关掉思考链。校对、列术语这类机械 JSON 任务不需要推理，
    而有些模型（实测 claude-deepseek-v4.1-flash）会一路想到 max_tokens
    耗尽、正文一个字都不吐。max_tokens 只有直连那条路认，见 _llm_sdk。
    """
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return _llm_api(system, user, max_tokens, think)
    return _llm_sdk(system, user, think)


def _llm_api(system: str, user, max_tokens: int, think: bool = True) -> str:
    import anthropic
    client = anthropic.Anthropic()
    # 思考链和正文共用 max_tokens 这一个预算。写文稿这种"输出跟输入一样长"的
    # 任务，思考一多正文就被截在半篇上 —— 退回无思考再试一次，比整篇丢掉强。
    for t in ([think, False] if think else [think]):
        with client.messages.stream(
            model=MODEL, max_tokens=max_tokens,
            thinking={"type": "adaptive" if t else "disabled"},
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as st:
            msg = st.get_final_message()
        if msg.stop_reason != "max_tokens":
            break
        print("  ! 输出被 max_tokens 截断，关掉思考链重试", file=sys.stderr)
    if msg.stop_reason == "refusal":
        cat = getattr(msg.stop_details, "category", None)
        sys.exit(f"模型拒绝了请求（{cat}）")
    if msg.stop_reason == "max_tokens":
        sys.exit("输出被 max_tokens 截断（关掉思考链还是不够），调大 max_tokens 或拆小输入")
    return "".join(b.text for b in msg.content if b.type == "text")


def _sdk_prompt(user):
    """字符串直接给；图文块要走流式输入 —— Agent SDK 只在这条路上收图。"""
    if isinstance(user, str):
        return user

    async def gen():
        yield {"type": "user", "message": {"role": "user", "content": user}}

    return gen()


def _llm_sdk(system: str, user, think: bool = True) -> str:
    """最后的兜底：Claude Agent SDK，用你已登录的额度。

    不给 system_prompt=preset 就是精简系统提示（不是 Claude Code 那套完整
    preset）；再把工具全禁掉、只留一轮、不加载 CLAUDE.md，拿它当纯文本模型用。
    模型名照传 V2T_MODEL，不认就落回 SDK 自己的默认值。

    ponytail: Agent SDK 没有 max_tokens（temperature/top_p 也没有），这条路上
    输出长度卡不住；要卡得死只能配 ANTHROPIC_AUTH_TOKEN 走直连。
    """
    from claude_agent_sdk import (
        ClaudeAgentOptions, ClaudeSDKError, ResultMessage, query,
    )

    opts = ClaudeAgentOptions(
        system_prompt=system,
        # 只认显式设的 V2T_MODEL；没设就别传，让 CLI 自己从配置/ANTHROPIC_MODEL
        # 里挑 —— 显式传一个网关专用模型名会被它当成 unrecognized_model 警告
        model=os.getenv("V2T_MODEL"),
        disallowed_tools=["*"],       # 全禁，别让它去 Read/Bash
        setting_sources=[],           # 不读 CLAUDE.md 和本地 settings
        max_turns=1,
        thinking={"type": "adaptive" if think else "disabled"},
    )

    async def go():
        # 让 query() 自己跑完：中途 break/return 会触发 SDK 内部生成器的
        # aclose()，它自己会报 "asynchronous generator is already running" 刷屏
        res = None
        async for m in query(prompt=_sdk_prompt(user), options=opts):
            if isinstance(m, ResultMessage):
                res = m
        return res

    try:
        res = asyncio.run(go())
    except ClaudeSDKError as e:
        # 没登录、没装 CLI、CLI 报错都从这走。CLI 出错时是先 yield 一个
        # is_error 的 ResultMessage，再抛 ResultError，所以这里是主要出口
        sys.exit(f"Claude Agent SDK 调用失败：{e}"[:400])
    if res is None:
        sys.exit("Claude Agent SDK 没返回结果")
    if res.subtype != "success" or res.is_error:
        sys.exit(f"Claude Agent SDK 调用失败（{res.subtype}）："
                 f"{res.errors or res.result or ''}"[:400])
    return res.result or ""


def json_array(s: str):
    """抠出回复里的 JSON 数组；不是数组就返回 None。"""
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
    arr = json_array(llm(GLOSSARY_SYS, f"课程标题：{title}\n\n开头文本：\n{sample}", 4000, think=False))
    return "\n".join(str(x) for x in arr) if arr else ""


def _clean_block(block, gloss):
    src = "\n".join(f"{i}. {s['text']}" for i, s in enumerate(block, 1))
    user = (f"术语表：\n{gloss}\n\n" if gloss else "") + \
           f"待校对片段（{len(block)} 条）：\n{src}"
    for _ in range(2):
        arr = json_array(llm(CLEAN_SYS, user, 8000, think=False))
        if arr and len(arr) == len(block):
            return [str(x or "").strip() for x in arr]
    print(f"  ! 一块校对失败（{len(block)} 条），保留原文", file=sys.stderr)
    return [s["text"] for s in block]


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
