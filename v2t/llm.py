"""统一的模型调用层。"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures as cf
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .subs import fix_cjk_punct

CHUNK = 40
WORKERS = 4
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"

_PROVIDER_ALIASES = {
    "gpt": "openai",
    "openai-api": "openai",
    "claude": "anthropic",
    "anthropic-api": "anthropic",
    "claude-agent": "claude-sdk",
    "codex-cli": "codex",
}
_PROVIDERS = {"auto", "openai", "anthropic", "codex", "claude-sdk"}


class LLMConfigError(RuntimeError):
    """模型后端配置错误。"""


def _fail(message: str):
    sys.exit(message)


def _codex_binary() -> str:
    return os.getenv("V2T_CODEX_BIN", "codex")


def _has_codex() -> bool:
    return shutil.which(_codex_binary()) is not None


def _configured_provider() -> str:
    raw = (os.getenv("V2T_PROVIDER") or
           os.getenv("V2T_LLM_PROVIDER") or "auto").strip().lower()
    raw = _PROVIDER_ALIASES.get(raw, raw)
    if raw not in _PROVIDERS:
        choices = ", ".join(sorted(_PROVIDERS | set(_PROVIDER_ALIASES)))
        raise LLMConfigError(f"未知的 V2T_PROVIDER={raw!r}，可选：{choices}")
    return raw


def selected_provider() -> str:
    """返回本次运行实际选择的后端，不发起网络请求。"""
    configured = _configured_provider()
    if configured != "auto":
        return configured
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_AUTH_TOKEN"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if _has_codex():
        return "codex"
    raise LLMConfigError(
        "没有检测到可用的模型后端。请设置 OPENAI_API_KEY 并使用 "
        "V2T_PROVIDER=openai，或安装/登录 Codex CLI 后使用 V2T_PROVIDER=codex；"
        "如需旧的 Claude SDK，请显式设置 V2T_PROVIDER=claude-sdk"
    )


def selected_model(provider: str | None = None) -> str | None:
    """返回后端对应的模型名；Codex 未指定时交给其本地配置。"""
    provider = provider or selected_provider()
    if os.getenv("V2T_MODEL"):
        return os.getenv("V2T_MODEL")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
    if provider == "codex":
        return os.getenv("CODEX_MODEL") or None
    return os.getenv("ANTHROPIC_MODEL") or None


def llm(system: str, user, max_tokens: int = 16000, think: bool = True) -> str:
    """调用已配置的模型后端，user 可以是字符串或 content blocks。"""
    try:
        provider = selected_provider()
    except LLMConfigError as e:
        _fail(str(e))
    if provider == "openai":
        return _llm_openai(system, user, max_tokens)
    if provider == "anthropic":
        return _llm_anthropic(system, user, max_tokens, think)
    if provider == "codex":
        return _llm_codex(system, user)
    return _llm_sdk(system, user, think)


def _compact_error(error: Exception) -> str:
    text = str(error).strip().replace("\n", " ")
    return text[-800:] if text else error.__class__.__name__


def _anthropic_model() -> str:
    return selected_model("anthropic") or DEFAULT_ANTHROPIC_MODEL


def _llm_anthropic(system: str, user, max_tokens: int, think: bool = True) -> str:
    """Anthropic API 路径。"""
    try:
        import anthropic
    except ImportError as e:
        _fail("当前后端需要 anthropic：请先运行 uv sync，或设置 V2T_PROVIDER=openai/codex")
        raise AssertionError from e
    client = anthropic.Anthropic()
    for t in ([think, False] if think else [think]):
        with client.messages.stream(
            model=_anthropic_model(), max_tokens=max_tokens,
            thinking={"type": "adaptive" if t else "disabled"},
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as st:
            msg = st.get_final_message()
        if msg.stop_reason != "max_tokens":
            break
        print("  ! Anthropic 输出被 max_tokens 截断，关掉思考链重试", file=sys.stderr)
    if msg.stop_reason == "refusal":
        cat = getattr(msg.stop_details, "category", None)
        _fail(f"模型拒绝了请求（{cat}）")
    if msg.stop_reason == "max_tokens":
        _fail("输出被 max_tokens 截断（关掉思考链还是不够），调大 max_tokens 或拆小输入")
    return "".join(b.text for b in msg.content if b.type == "text")


def _llm_api(system: str, user, max_tokens: int, think: bool = True) -> str:
    """兼容旧代码的 Anthropic API 名称。"""
    return _llm_anthropic(system, user, max_tokens, think)


def _openai_input(user):
    """把现有 content blocks 转成 Responses API 的 input。"""
    if isinstance(user, str):
        blocks = [{"type": "input_text", "text": user}]
    else:
        blocks = []
        for block in user:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                blocks.append({
                    "type": "input_text",
                    "text": str(block.get("text", "")),
                })
                continue
            if kind == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    url = f"data:{media_type};base64,{source.get('data', '')}"
                elif source.get("url"):
                    url = source["url"]
                else:
                    raise LLMConfigError("图片块缺少 base64 data 或 url")
                blocks.append({
                    "type": "input_image",
                    "image_url": url,
                    "detail": os.getenv("V2T_IMAGE_DETAIL", "high"),
                })
                continue
            if kind == "image_url":
                image_url = block.get("image_url") or {}
                if not isinstance(image_url, dict) or not image_url.get("url"):
                    raise LLMConfigError("image_url 图片块缺少 url")
                blocks.append({
                    "type": "input_image",
                    "image_url": image_url["url"],
                    "detail": image_url.get(
                        "detail", os.getenv("V2T_IMAGE_DETAIL", "high")
                    ),
                })
                continue
            raise LLMConfigError(f"OpenAI 后端不认识 content block 类型：{kind!r}")

    return [
        {"role": "user", "content": blocks},
    ]


def _response_field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _openai_result_text(response) -> str:
    """从 Responses 对象取可见文本，并识别拒答。"""
    refusal = None
    for item in _response_field(response, "output", []) or []:
        for content in _response_field(item, "content", []) or []:
            kind = _response_field(content, "type")
            if kind == "refusal":
                refusal = _response_field(content, "refusal", "")
    if refusal:
        _fail(f"OpenAI 模型拒绝了请求：{refusal}")

    text = _response_field(response, "output_text", "")
    if callable(text):
        text = text()
    return str(text or "")


def _llm_openai(system: str, user, max_tokens: int) -> str:
    """OpenAI Responses API 路径，支持文本和 base64 图片。"""
    try:
        from openai import OpenAI
    except ImportError as e:
        _fail("当前后端需要 openai：请先运行 uv sync")
        raise AssertionError from e

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_AUTH_TOKEN")
    if not api_key:
        _fail("V2T_PROVIDER=openai 需要 OPENAI_API_KEY（或 OPENAI_AUTH_TOKEN）")
    kwargs = {"api_key": api_key}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    if os.getenv("OPENAI_ORG_ID"):
        kwargs["organization"] = os.getenv("OPENAI_ORG_ID")
    if os.getenv("OPENAI_PROJECT_ID"):
        kwargs["project"] = os.getenv("OPENAI_PROJECT_ID")
    client = OpenAI(**kwargs)

    request = {
        "model": selected_model("openai") or DEFAULT_OPENAI_MODEL,
        "instructions": system,
        "input": _openai_input(user),
        "max_output_tokens": max_tokens,
    }
    try:
        response = client.responses.create(**request)
    except Exception as e:
        _fail(f"OpenAI Responses API 调用失败：{_compact_error(e)}")

    status = _response_field(response, "status")
    if status == "incomplete":
        details = _response_field(response, "incomplete_details")
        reason = _response_field(details, "reason")
        suffix = f"（原因：{reason}）" if reason else ""
        _fail(f"OpenAI 输出未完成{suffix}，请调大 max_output_tokens 或拆小输入")
    if status in {"failed", "cancelled"}:
        error = _response_field(response, "error")
        message = _response_field(error, "message") or _response_field(error, "code")
        _fail(f"OpenAI Responses API 返回 {status}：{message or '未知错误'}")
    text = _openai_result_text(response).strip()
    if not text:
        _fail("OpenAI Responses API 返回了空文本")
    return text


def _data_url_parts(url: str):
    """解析 data URL，返回 (media_type, bytes)，普通 URL 返回 None。"""
    if not url.startswith("data:") or "," not in url:
        return None
    header, encoded = url.split(",", 1)
    media_type = header[5:].split(";", 1)[0] or "image/jpeg"
    try:
        return media_type, base64.b64decode(encoded)
    except (ValueError, base64.binascii.Error) as e:
        raise LLMConfigError("图片 data URL 不是有效的 base64") from e


def _codex_image_bytes(block):
    kind = block.get("type") if isinstance(block, dict) else None
    if kind == "image":
        source = block.get("source") or {}
        if source.get("type") == "base64":
            return source.get("media_type", "image/jpeg"), base64.b64decode(source.get("data", ""))
        if source.get("url"):
            return _data_url_parts(source["url"])
    if kind == "image_url":
        image_url = block.get("image_url") or {}
        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
        return _data_url_parts(url)
    return None


def _codex_text(user) -> str:
    if isinstance(user, str):
        return user
    return "\n\n".join(
        str(block.get("text", ""))
        for block in user
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _llm_codex(system: str, user) -> str:
    """通过官方 Codex CLI 的非交互模式调用本机已登录的 Codex。"""
    binary = _codex_binary()
    executable = shutil.which(binary)
    if executable is None:
        _fail(
            f"找不到 Codex CLI：{binary!r} 不在 PATH。"
            "请安装并登录 Codex，或改用 V2T_PROVIDER=openai"
        )

    prompt = (
        f"{system}\n\n"
        "--- 用户输入 ---\n"
        f"{_codex_text(user)}\n\n"
        "请只返回最终答案，不要修改文件、运行命令或解释你的工具调用。"
    )
    timeout_raw = os.getenv("V2T_CODEX_TIMEOUT", "900")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 900.0

    with tempfile.TemporaryDirectory(prefix="v2t-codex-") as tmp:
        tmp_path = Path(tmp)
        output = tmp_path / "answer.txt"
        image_paths = []
        for i, block in enumerate(user if isinstance(user, list) else []):
            if not isinstance(block, dict) or block.get("type") not in {"image", "image_url"}:
                continue
            parts = _codex_image_bytes(block)
            if parts is None:
                _fail("Codex CLI 只能接收本地图片或 base64 图片块，无法直接接收远程图片 URL")
            media_type, data = parts
            suffix = mimetypes.guess_extension(media_type) or ".img"
            path = tmp_path / f"image-{i}{suffix}"
            path.write_bytes(data)
            image_paths.append(path)

        cmd = [
            executable, "exec",
            "--ephemeral",
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--color", "never",
            "--output-last-message", str(output),
        ]
        model = selected_model("codex")
        if model:
            cmd.extend(["--model", model])
        for path in image_paths:
            cmd.extend(["--image", str(path)])
        cmd.append("-")

        try:
            result = subprocess.run(
                cmd, input=prompt, text=True, capture_output=True,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            _fail(f"Codex CLI 超时（{timeout:g}s），可调大 V2T_CODEX_TIMEOUT")
        except OSError as e:
            _fail(f"启动 Codex CLI 失败：{_compact_error(e)}")

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            _fail(f"Codex CLI 调用失败（退出码 {result.returncode}）：{detail[-1000:]}")
        answer = output.read_text(errors="replace") if output.exists() else result.stdout
        if not answer.strip():
            _fail("Codex CLI 没有返回最终答案")
        return answer.strip()


def _sdk_prompt(user):
    """字符串直接给；图文块要走流式输入，Agent SDK 只在这条路上收图。"""
    if isinstance(user, str):
        return user

    async def gen():
        yield {"type": "user", "message": {"role": "user", "content": user}}

    return gen()


def _llm_sdk(system: str, user, think: bool = True) -> str:
    """兼容旧配置：使用本机已登录的 Claude Agent SDK。"""
    try:
        from claude_agent_sdk import (
            ClaudeAgentOptions, ClaudeSDKError, ResultMessage, query,
        )
    except ImportError as e:
        _fail(
            "没有可用的模型后端。请设置 OPENAI_API_KEY 并使用 V2T_PROVIDER=openai，"
            "或安装/登录 Codex CLI 后使用 V2T_PROVIDER=codex"
        )
        raise AssertionError from e

    opts = ClaudeAgentOptions(
        system_prompt=system,
        model=os.getenv("V2T_MODEL"),
        disallowed_tools=["*"],
        setting_sources=[],
        max_turns=1,
        thinking={"type": "adaptive" if think else "disabled"},
    )

    async def go():
        res = None
        async for m in query(prompt=_sdk_prompt(user), options=opts):
            if isinstance(m, ResultMessage):
                res = m
        return res

    try:
        res = asyncio.run(go())
    except ClaudeSDKError as e:
        _fail(f"Claude Agent SDK 调用失败：{e}"[:400])
    if res is None:
        _fail("Claude Agent SDK 没返回结果")
    if res.subtype != "success" or res.is_error:
        _fail(f"Claude Agent SDK 调用失败（{res.subtype}）："
              f"{res.errors or res.result or ''}"[:400])
    return res.result or ""


def json_array(s: str):
    """抠出回复里的 JSON 数组；不是数组就返回 None。"""
    s = re.sub(r"^" + chr(96) * 3 + r"(?:json)?\s*|" +
               chr(96) * 3 + r"$", "", s.strip())
    i, j = s.find("[") , s.rfind("]")
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
    arr = json_array(llm(GLOSSARY_SYS,
                         f"课程标题：{title}\n\n开头文本：\n{sample}",
                         4000, think=False))
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
