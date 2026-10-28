#!/bin/bash
# webui 常驻启动：在你自己的终端里跑一次（或加进 AutoDL 开机脚本），
# 不依赖任何开发会话的存活。日志在 logs/webui_stdout.log。
# 已在跑则直接退出（幂等，可重复执行/用作保活）。
cd "$(dirname "$0")"
if pgrep -f "src/webui/app.py" >/dev/null; then
  echo "webui 已在运行: http://localhost:6006"
  exit 0
fi
mkdir -p logs
setsid nohup python src/webui/app.py > logs/webui_stdout.log 2>&1 < /dev/null &
disown
sleep 3
if curl -s --noproxy '*' -o /dev/null -w '%{http_code}' http://localhost:6006/ | grep -q 200; then
  echo "webui 已启动: http://localhost:6006"
else
  echo "启动中…几秒后用浏览器访问 :6006；若异常查看 logs/webui_stdout.log"
fi
