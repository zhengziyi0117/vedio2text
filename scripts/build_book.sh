#!/usr/bin/env bash
# 把 work/*/course.md 收成一棵 mdBook 源码树 → book_src/
# 新增课程目录后重跑即可，SUMMARY.md 自动生成。
set -euo pipefail
cd "$(dirname "$0")/.."

src=book_src
rm -rf "$src"
mkdir -p "$src"
: > "$src/SUMMARY.md"

for d in work/*/; do
  [ -f "$d/course.md" ] || continue
  name=$(basename "$d")
  mkdir -p "$src/$name"
  cp "$d/course.md" "$src/$name/"
  # course.md 里配图写的是 assets/xxx.jpg 这种相对路径，得跟着搬
  [ -d "$d/assets" ] && cp -R "$d/assets" "$src/$name/"

  # 章节名取正文第一个 H1，没有就用目录名
  title=$(grep -m1 '^# ' "$d/course.md" || true)
  title=${title#\# }
  printf -- '- [%s](%s/course.md)\n' "${title:-$name}" "$name" >> "$src/SUMMARY.md"
done

echo "book_src/ 就绪：$(grep -c '^- \[' "$src/SUMMARY.md") 章"
