@echo off
echo Setting up virtual environment for handwriting_gradio_app...

python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Setup complete! You can now run the app with:
echo python app.py
echo.
pause
