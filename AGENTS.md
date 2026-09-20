# AGENTS.md

课程视频 → 字幕 + 可读文稿。跑法、参数、环境变量看 `README.md`，这里只记改代码时需要知道的事。

## 跑与自检

```bash
uv run -m v2t "<视频文件或链接>"        # 下载 → 转写 → 校对 → 写稿
uv run -m v2t --selftest               # 纯函数自检，不联网
uv run -m v2t --llm-test               # 用当前后端发一次最小请求
```

改了 `subs.py` / `media.py` / `doc.py` 里的纯函数（解析、时间戳、归并、标点、段落标记）就
往 `selftest.py` 补一条断言 —— 那里面每条都是踩过的坑。改 prompt 不用重转写：

```bash
uv run -m v2t <src> --from clean   # 重跑校对 + 写稿
uv run -m v2t <src> --from doc     # 只重跑写稿
```

产物在 `work/<课程>/`，已存在的步骤产物会自动跳过，删掉对应文件即可强制重跑。

## 字幕从哪来

优先级：yt-dlp 下载的字幕 → 同名外挂字幕 → 内嵌字幕 → whisper 转写。

**质量上 whisper（mlx large-v3-turbo）通常比现成字幕准**：B 站 AI 字幕（`ai-zh`）、YouTube
机翻轨普遍没标点、口吃重复多、有同音错字；whisper 带标点、断句正常。现成字幕的唯一优势是快
（秒级 vs 98 分钟音频约 8 分钟）。所以：

- 默认「有现成字幕就复用」，省时间；
- 要质量就 `V2T_FORCE_ASR=1` 强制转写（无损，只是慢）；
- 拿不准时两条都跑一遍对比：先默认跑一次留 `subs.json`，再 `V2T_FORCE_ASR=1` 跑一次，
  比 `transcript.srt`。
- 字幕糙不要紧：文稿那步（`doc.py` 的 `DOC_SYS`）负责把没标点/口吃重复整理成通顺段落，
  原始字幕原样留在 `subs.json` / `transcript.srt` 里。别去改原始字幕。

字幕是「有则省事、无则 ASR」的降级路径，**取不到不能把整条流程带走**：所有取字幕的调用都
`check=False`，失败只打印原因然后回去跑 ASR。首选语种一条都没有时还会退一步抓任意语种
（日语课只挂 `ja` 轨也比 ASR 强），文稿那步按 `V2T_DOC_LANG` 翻译。

## 踩过的坑（改这块之前先看一眼）

- `yt-dlp --sub-langs` 是**正则 fullmatch**，不是 glob。B 站的机翻码是 `ai-zh`，得写 `ai-.*`
  （写 `ai-*` 匹配不到，而且 yt-dlp 返回码还是 0、stderr 为空，静默一条字幕都不下）。
  语种没匹配上时那句提示在 **stdout** 里，不在 stderr。
- WebVTT 允许省掉小时位（`00:00.920 --> 00:03.000`，ffmpeg 转出来的字幕一小时内都长这样）。
  时间戳正则只认 `H:MM:SS` 会把前半段 cue 静默丢光。
- yt-dlp 对 B 站合集的 `--flat-playlist` 只给 URL 不给标题，`--list` 列出来标题是 `NA`。
- B 站的 CC/AI 字幕、会员清晰度要登录态（`V2T_COOKIES`）；macOS 上 Safari 要终端有
  Full Disk Access，Arc 不在 yt-dlp 支持列表里（解密的钥匙串名对不上）。

## 约定

- 不引新依赖：标准库、已装的 `yt-dlp` / `ffmpeg` 优先。
- 模型后端不写死，一律走 `llm.py` 的 `llm()`；不要在新代码里直接调某个厂商 SDK。
- 外部输入（模型输出、字幕文件、yt-dlp 结果）都要当成不可信：越界、缺字段、空值一律丢，
  别让它脏到主流程。
