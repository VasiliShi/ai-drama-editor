#!/bin/bash
# 将所有 *_landscape.mp4 和 *_portrait.mp4 收集到 output/results/ 目录
# 用法: bash collect_results.sh

set -e

RESULTS_DIR="output/results"
mkdir -p "$RESULTS_DIR"

count=0
for f in output/*/*_landscape.mp4 output/*/*_portrait.mp4; do
  [ -f "$f" ] || continue
  mv "$f" "$RESULTS_DIR/"
  count=$((count + 1))
done

echo "已移动 $count 个文件到 $RESULTS_DIR/"
ls -1 "$RESULTS_DIR/"