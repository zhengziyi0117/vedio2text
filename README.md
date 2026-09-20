# video-2-text

课程视频 → 字幕 + 可读文稿。本地 whisper/现成字幕转写，支持 Anthropic、OpenAI GPT、Codex CLI 等模型后端，可选抽帧配图。

产出的文稿自动汇总成在线书：<https://zhengziyi0117.github.io/vedio2text/>

## 装环境

模型后端不是写死的。最简单的 GPT API 配置：

    export V2T_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    export OPENAI_MODEL=gpt-4o-mini

如果本机已经登录 Codex CLI，可以不配置 API key：

    export V2T_PROVIDER=codex
    uv run -m v2t --llm-test

没有设置 V2T_PROVIDER 时，auto 会按 OpenAI API、Anthropic API、Codex CLI 的顺序选择；都没有时明确报错，不会暗中调用 Claude。

需要 macOS（Apple Silicon）+ Python 3.10+。转写走 MLX，Intel Mac / Linux 跑不了。

```bash
brew install ffmpeg yt-dlp

uv sync && source .venv/bin/activate
```

依赖写在 `pyproject.toml`，`uv sync` 会照 `uv.lock` 建好 `.venv`（`mlx-whisper` 带 Apple Silicon 判断，别的平台自动跳过）。之后跑脚本一律 `uv run -m v2t ...`，不用先 activate。

模型后端通过统一的 `llm()` 接口选择：

1. 设置 `V2T_PROVIDER=openai`，通过 OpenAI Responses API 调用 GPT；支持 `OPENAI_BASE_URL`，也支持当前抽帧挑图使用的 base64 图片输入。
2. 设置 `V2T_PROVIDER=codex`，调用本机 `codex exec` 非交互模式；文本和图片都会转成 Codex CLI 可接收的输入。
3. `V2T_PROVIDER=anthropic` 保留原来的 Anthropic API；`claude-sdk` 仍可显式使用本机 Claude Code。
4. 默认 `auto` 根据已配置的 key/CLI 自动选择；如果都没有，必须显式配置后端。想先验证配置，运行 `uv run -m v2t --llm-test`。

模型名优先取 `V2T_MODEL`，然后按后端取 `OPENAI_MODEL`、`CODEX_MODEL` 或 `ANTHROPIC_MODEL`。

模型权重可选：不手动拉的话，第一次转写会自动从 HuggingFace 下 ~1.6GB（本机 hf-xet 会卡死，见下）。

```bash
./scripts/fetch_model.sh mlx-community/whisper-large-v3-turbo
```

> 为什么不直接让 mlx_whisper 自己下：本机 `huggingface-hub 1.31.0 + hf-xet 1.6.0` 下 safetensors 会创建 0 字节 `.incomplete` 后永久卡死，`HF_HUB_DISABLE_XET=1` 也没用。`fetch_model.sh` 绕开 hf_hub 用 curl 分段拉，脚本会检测到 `models/<名字>/` 并优先用本地权重。

## 跑

给一个视频文件或 URL 就行，三步一条龙：

```bash
uv run -m v2t ~/Downloads/讲座.mp4
uv run -m v2t "https://www.youtube.com/watch?v=xxxxxxxx"
```

整个播放列表也吃，用 `--list` 看编号、`--ep` 挑一集（一个 work 目录只装一集，别整个列表丢进去）：

```bash
uv run -m v2t "<带 list= 的链接>" --list     # 列出分集和编号
uv run -m v2t "<带 list= 的链接>" --ep 3     # 只处理第 3 集
```

产物落在 `work/<课程名>/`：

| 文件 | 说明 |
| --- | --- |
| `subs.json` | 原始字幕（whisper 转写，或 yt-dlp/外挂/内嵌字幕复用） |
| `clean.json` | 当前模型后端校对后的字幕 |
| `transcript.srt` | 校对后的 srt，可直接挂播放器 |
| `course.md` | 最终文稿，每个自然段末尾带可跳回视频的时间链接 |
| `assets/` | `--shots` 时的配图 |

一门课分多讲时加 `--series`，产物归到 `work/<系列名>/<讲名>/`，出书时这门课自成一组：

```bash
uv run -m v2t "<第一讲链接>" --series CS336
uv run -m v2t "<第二讲链接>" --series CS336   # 系列同名就排在一起
```

## 常用参数

```bash
uv run -m v2t <src> --shots          # 另下 720p 视频，抽帧给文稿配图（多约 130MB）
uv run -m v2t <src> --only subs      # 只出字幕，不调 LLM
uv run -m v2t <src> --from clean     # 跳过下载和转写，只重跑校对+写稿
uv run -m v2t <src> --from doc       # 只重跑写稿
uv run -m v2t <src> --work /tmp/w    # 换产物目录（默认 work/）
uv run -m v2t <src> --series CS336   # 归到某门课下面（work/CS336/<讲名>/）
uv run -m v2t --selftest             # 解析器自检，不联网
uv run -m v2t --llm-test             # 用当前后端发送一次真实最小请求
```

段落时间链接不用 `--shots` 也有，每篇文稿都会生成。源是本地文件时没有地址可跳，只显示 `*12:34*` 这样的纯时间。

改完 prompt 想重跑不用重转写，`--from clean` 或 `--from doc` 直接吃已有中间产物。已存在的步骤产物会自动跳过，删掉对应文件即可强制重跑。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `V2T_PROVIDER` / `V2T_LLM_PROVIDER` | `auto` | `auto`、`openai`、`codex`、`anthropic` 或 `claude-sdk` |
| `OPENAI_API_KEY` | — | 使用 OpenAI GPT API 时必填 |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI 后端默认模型 |
| `OPENAI_BASE_URL` | OpenAI 默认地址 | 可选的 OpenAI 兼容网关地址 |
| `V2T_IMAGE_DETAIL` | `high` | GPT 视觉模型接收配图时的细节级别 |
| `V2T_MODEL` | 按后端决定 | 统一覆盖当前后端的模型名 |
| `CODEX_MODEL` | Codex 本地配置 | `V2T_PROVIDER=codex` 时可选 |
| `V2T_CODEX_BIN` | `codex` | Codex CLI 可执行文件名或路径 |
| `V2T_CODEX_TIMEOUT` | `900` | Codex CLI 单次请求超时秒数 |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | — | 保留的 Anthropic API 配置 |
| `ANTHROPIC_BASE_URL` | — | Anthropic 自建网关地址 |
| `V2T_ASR_MODEL` | `mlx-community/whisper-large-v3-turbo` | HF repo，或 `models/` 下的目录名 |
| `V2T_LANG` | 自动检测 | 源语言。设了它，抢字幕时该语言优先于 `LANG_PREF` 里的中文默认值 |
| `V2T_DOC_LANG` | `中文` | 文稿写成什么语言 |
| `V2T_RATE_LIMIT` | `2M` | 下载限速，跑满带宽容易招 429；`0` 为不限速 |
| `V2T_COOKIES` | — | 需要登录的站点（B 站的 CC 字幕等）：浏览器名 `chrome`/`safari`，或 cookies.txt 路径 |
| `V2T_FORCE_ASR` | — | 置 `1` 无视现成字幕、强制 whisper 转写。现成字幕快但常是机翻/没标点，whisper 通常更准 |

## 出书

推到 `main` 就自动重建，不用手动跑。本地预览：

```bash
./scripts/build_book.sh   # work/ 下的文稿 + assets/ → book_src/
mdbook build              # → book/，开 book/index.html
```

`scripts/build_book.sh` 每次重新扫 `work/`，新存的课程自动进目录。`work/<课程>/` 单独成章；`work/<系列>/<课程>/` 收成子目录，系列名当目录名（想写课程简介就在 `work/<系列>/` 放个 `README.md`，没有就自动生成一张标题页）。章节名取正文第一个一级标题，没有就用目录名；同一系列内按目录名排序，所以 `...Lecture 2...` 排在 `...Lecture 10...` 前面。想本地装 mdBook：`brew install mdbook`。

## 代码结构

```
v2t/
  subs.py      字幕文本层：SRT/VTT 解析、句子级归并、时间戳、中文标点
  media.py     取材：yt-dlp 下载、现成/外挂/内嵌字幕、whisper 转写
  llm.py       统一调模型：OpenAI / Codex / Anthropic / Claude SDK，外加字幕校对
  doc.py       文稿：写讲稿、回填时间链接、抽帧挑图挂进段落
  cli.py       命令行：参数、选集、按步骤跑
  selftest.py  --selftest 的纯函数自检
```

数据一路向上走：`media` 出 `[{start, end, text}]` → `llm` 校对 → `doc` 出 markdown。
每步的产物都落在 `work/<课程名>/`，中断了用 `--from` 接着跑。
