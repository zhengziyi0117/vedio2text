# video-2-text

课程视频 → 字幕 + 可读文稿。本地 whisper 转写，Claude 校对并整理成 markdown 讲稿，可选抽帧配图。

产出的文稿自动汇总成在线书：<https://zhengziyi0117.github.io/vedio2text/>

## 装环境

需要 macOS（Apple Silicon）+ Python 3.10+。转写走 MLX，Intel Mac / Linux 跑不了。

```bash
brew install ffmpeg yt-dlp

uv sync && source .venv/bin/activate
```

依赖写在 `pyproject.toml`，`uv sync` 会照 `uv.lock` 建好 `.venv`（`mlx-whisper` 带 Apple Silicon 判断，别的平台自动跳过）。之后跑脚本一律 `uv run v2t.py ...`，不用先 activate。

模型不用配。脚本按这个顺序找：

1. 环境里有 `ANTHROPIC_API_KEY` 或 `ANTHROPIC_AUTH_TOKEN` → 直连，`ANTHROPIC_BASE_URL` 一并生效（自建网关只配 auth token 也能用）
2. 都没有 → 回退到 Claude Agent SDK（底层就是本机 Claude Code，用它已登录的额度）

模型名取 `V2T_MODEL`，没设就跟随 `ANTHROPIC_MODEL`，再没有才落回 `claude-opus-5`。回退那条路只在显式设了 `V2T_MODEL` 时才把模型名传下去，否则交给 Claude Code 自己挑。

模型权重可选：不手动拉的话，第一次转写会自动从 HuggingFace 下 ~1.6GB（本机 hf-xet 会卡死，见下）。

```bash
./scripts/fetch_model.sh mlx-community/whisper-large-v3-turbo
```

> 为什么不直接让 mlx_whisper 自己下：本机 `huggingface-hub 1.31.0 + hf-xet 1.6.0` 下 safetensors 会创建 0 字节 `.incomplete` 后永久卡死，`HF_HUB_DISABLE_XET=1` 也没用。`fetch_model.sh` 绕开 hf_hub 用 curl 分段拉，脚本会检测到 `models/<名字>/` 并优先用本地权重。

## 跑

给一个视频文件或 URL 就行，三步一条龙：

```bash
uv run v2t.py ~/Downloads/讲座.mp4
uv run v2t.py "https://www.youtube.com/watch?v=xxxxxxxx"
```

整个播放列表也吃，用 `--list` 看编号、`--ep` 挑一集（一个 work 目录只装一集，别整个列表丢进去）：

```bash
uv run v2t.py "<带 list= 的链接>" --list     # 列出分集和编号
uv run v2t.py "<带 list= 的链接>" --ep 3     # 只处理第 3 集
```

产物落在 `work/<课程名>/`：

| 文件 | 说明 |
| --- | --- |
| `subs.json` | 原始字幕（whisper 转写，或 yt-dlp/外挂/内嵌字幕复用） |
| `clean.json` | Claude 校对后的字幕 |
| `transcript.srt` | 校对后的 srt，可直接挂播放器 |
| `course.md` | 最终文稿，每个自然段末尾带可跳回视频的时间链接 |
| `assets/` | `--shots` 时的配图 |

## 常用参数

```bash
uv run v2t.py <src> --shots          # 另下 720p 视频，抽帧给文稿配图（多约 130MB）
uv run v2t.py <src> --only subs      # 只出字幕，不调 LLM
uv run v2t.py <src> --from clean     # 跳过下载和转写，只重跑校对+写稿
uv run v2t.py <src> --from doc       # 只重跑写稿
uv run v2t.py <src> --work /tmp/w    # 换产物目录（默认 work/）
uv run v2t.py --selftest             # 解析器自检，不联网
```

段落时间链接不用 `--shots` 也有，每篇文稿都会生成。源是本地文件时没有地址可跳，只显示 `*12:34*` 这样的纯时间。

改完 prompt 想重跑不用重转写，`--from clean` 或 `--from doc` 直接吃已有中间产物。已存在的步骤产物会自动跳过，删掉对应文件即可强制重跑。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | — | 二选一即可，都没有才走 Claude Agent SDK（本机 Claude Code） |
| `ANTHROPIC_BASE_URL` | — | 走自建网关时 SDK 会自动认 |
| `V2T_MODEL` | 跟随 `ANTHROPIC_MODEL` | 不设就用你 Claude Code 里配的模型，最后才落 `claude-opus-5` |
| `V2T_ASR_MODEL` | `mlx-community/whisper-large-v3-turbo` | HF repo，或 `models/` 下的目录名 |
| `V2T_LANG` | 自动检测 | 源语言。设了它，抢字幕时该语言优先于 `LANG_PREF` 里的中文默认值 |
| `V2T_DOC_LANG` | `中文` | 文稿写成什么语言 |
| `V2T_RATE_LIMIT` | `2M` | 下载限速，跑满带宽容易招 429；`0` 为不限速 |

## 出书

推到 `main` 就自动重建，不用手动跑。本地预览：

```bash
./scripts/build_book.sh   # work/*/course.md + assets/ → book_src/
mdbook build              # → book/，开 book/index.html
```

`scripts/build_book.sh` 每次重新扫 `work/`，新存的课程目录自动进目录（章节名取正文第一个一级标题，没有就用目录名）。想本地装 mdBook：`brew install mdbook`。
