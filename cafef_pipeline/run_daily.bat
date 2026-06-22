@echo off
REM ── Chạy pipeline daily bằng Ollama local (cho Windows Task Scheduler) ──
REM Yêu cầu: Ollama đang chạy (app khay hệ thống / ollama serve).
cd /d E:\News-driven-stock\cafef_pipeline
"C:\Users\Admin\anaconda3\envs\tf-gpu\python.exe" run_daily.py >> daily_run.log 2>&1
