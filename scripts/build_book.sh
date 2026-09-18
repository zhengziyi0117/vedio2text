#!/usr/bin/env bash
# 把 work/*/course.md 收成一棵 mdBook 源码树 → book_src/
# work/<课程>/ 是单章；work/<系列>/<课程>/ 收成子目录（多讲一门课用这个）。
# 新增课程/系列后重跑即可，SUMMARY.md 自动生成。
set -euo pipefail
cd "$(dirname "$0")/.."

src=book_src
rm -rf "$src"
mkdir -p "$src"
: > "$src/SUMMARY.md"

# 讲次序：把目录名里的 "Lecture N" 抠出来当主键（Lecture 10 要排在 Lecture 2 后面），
# 没有讲次号的按名字排。不用 sort -V：整条长路径下 BSD 与 GNU 给出的顺序不一致，
# 实测 macOS 上 "...-2026-Le" 排在 "...-2026-Lecture-2-..." 后面，讲次就反了。
dirs() {
  local d n
  for d in "${1%/}"/*/; do
    n=$(printf '%s' "$d" | grep -oiE 'lecture[-_ ]?[0-9]+' | head -1 | grep -oE '[0-9]+' || true)
    printf '%010d\t%s\n' "${n:-0}" "$d"
  done | sort | cut -f2-
}

# 收一章：$1 源目录，$2 book_src 里的相对路径，$3 缩进
chapter() {
  local title
  mkdir -p "$src/$2"
  cp "$1/course.md" "$src/$2/"
  # course.md 里配图写的是 assets/xxx.jpg 这种相对路径，得跟着搬
  [ -d "$1/assets" ] && cp -R "$1/assets" "$src/$2/"

  # 章节名取正文第一个 H1，没有就用目录名
  title=$(grep -m1 '^# ' "$1/course.md" || true)
  title=${title#\# }
  printf -- '%s- [%s](%s/course.md)\n' "$3" "${title:-$(basename "$1")}" "$2" >> "$src/SUMMARY.md"
}

for d in $(dirs work); do
  name=$(basename "$d")
  if [ -f "$d/course.md" ]; then
    chapter "$d" "$name" ""
    continue
  fi

  # 系列目录：下面一讲都没有就不算系列
  n=$(ls -d "$d"*/course.md 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then
    continue
  fi

  # 系列自己那页：work/<系列>/README.md 有就用（写课程简介），没有就生成个标题页
  mkdir -p "$src/$name"
  if [ -f "$d/README.md" ]; then
    cp "$d/README.md" "$src/$name/"
  else
    printf '# %s\n' "$name" > "$src/$name/README.md"
  fi
  printf -- '- [%s](%s/README.md)\n' "$name" "$name" >> "$src/SUMMARY.md"
  for k in $(dirs "$d"); do
    if [ -f "$k/course.md" ]; then
      chapter "$k" "$name/$(basename "$k")" "  "
    fi
  done
done

echo "book_src/ 就绪：$(grep -c '^- \[' "$src/SUMMARY.md") 章"
