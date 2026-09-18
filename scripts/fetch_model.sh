#!/bin/bash
# 从 HuggingFace 拉 mlx 模型到 models/<name>/。
#
# 为什么不直接用 mlx_whisper(path_or_hf_repo="...")：
# 本机 huggingface-hub 1.31.0 + hf-xet 1.6.0 下载 safetensors 会创建 0 字节的
# .incomplete 后永久卡死（小 json 正常），HF_HUB_DISABLE_XET=1 无效。
# curl 拉同一个 CDN 正常，所以绕开 hf_hub 自己下。
#
# 单连接会被限速到 ~260KB/s（1.6GB 要 100 分钟），故分段并行。
# 但也不该跑满带宽 —— 总速率由 V2T_RATE_LIMIT_KB 控制，平均分给每段。
#
#   ./scripts/fetch_model.sh mlx-community/whisper-large-v3-turbo [段数=4]
set -euo pipefail

REPO="${1:?用法: $0 <repo_id> [段数]}"
N="${2:-4}"
RATE_KB="${V2T_RATE_LIMIT_KB:-1024}"   # 总速率预算，KB/s；设 0 不限速
NAME="${REPO##*/}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/models/$NAME"
BASE="https://huggingface.co/$REPO/resolve/main"

mkdir -p "$OUT"
echo "→ $REPO  →  $OUT"
curl -fsSL -o "$OUT/config.json" "$BASE/config.json"

URL="$BASE/weights.safetensors"
SIZE=$(curl -sIL "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}')
[ "$SIZE" -gt 0 ] || { echo "拿不到文件大小，检查 repo 名" >&2; exit 1; }

LIMIT=""
if [ "$RATE_KB" -gt 0 ]; then
    PER=$(( RATE_KB / N )); [ "$PER" -lt 1 ] && PER=1
    LIMIT="--limit-rate ${PER}K"
    echo "  总大小 $((SIZE / 1048576)) MB，分 $N 段并行，限速 ${PER}K/s×$N ≈ $((PER * N))K/s"
else
    echo "  总大小 $((SIZE / 1048576)) MB，分 $N 段并行，不限速"
fi

# $LIMIT 故意不加引号：为空时展开成零个参数（macOS 自带 bash 3.2 下空数组 + set -u 会报错）
for i in $(seq 0 $((N - 1))); do
    curl -fsSL $LIMIT -r "$((i * SIZE / N))-$(((i + 1) * SIZE / N - 1))" -o "$OUT/.part$i" "$URL" &
done
wait

for i in $(seq 0 $((N - 1))); do cat "$OUT/.part$i"; done > "$OUT/weights.safetensors"
rm -f "$OUT"/.part*

GOT=$(wc -c < "$OUT/weights.safetensors" | tr -d ' ')
[ "$GOT" = "$SIZE" ] || { echo "大小不符: 期望 $SIZE 实得 $GOT" >&2; exit 1; }
echo "  完成 $(du -h "$OUT/weights.safetensors" | cut -f1)"
