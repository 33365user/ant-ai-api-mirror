echo "This will run the backend. Continue (ENTER)?"
pause

"%cd%\interpreter\Scripts\waitress-serve.exe" --port=5000 app:app
echo "Backend Service Completed (Fatal or Normal Exit)"
pause