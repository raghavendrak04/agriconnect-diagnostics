@echo off
echo Installing dependencies if not already installed...
pip install streamlit tensorflow pillow
echo.
echo Starting the App...
streamlit run main.py
pause
