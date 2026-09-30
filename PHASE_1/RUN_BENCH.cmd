@echo off
REM Run from Explorer or Cursor integrated terminal — NOT from Agent stub shell.
REM Does NOT use DeepSeek. Fails if agent.api_key empty unless you set another provider.
cd /d "%~dp0"
set PYTHONPATH=src
echo === dabench status ===
uv run dabench status --config configs\react_baseline.yaml
echo.
echo === If you want Cursor-agent offline (no LLM API): not supported by dabench ReAct. ===
echo === Use a real API provider in react_baseline.yaml, OR run SQL packs manually. ===
echo.
echo Full bench:
echo   uv run dabench run-benchmark --config configs\react_baseline.yaml
pause
