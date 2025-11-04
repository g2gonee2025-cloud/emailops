@echo off
echo.
echo === EmailOps Offline Export Launcher ===
echo.
echo Starting EmailOps export script...
echo Output folder: %USERPROFILE%\Desktop\EmailExports
echo.
echo Note: A folder picker will open. Select the Outlook folders you want to export.
echo Press Ctrl+C to cancel at any time.
echo.
powershell.exe -ExecutionPolicy Bypass -File EmailOps.Offline.Export.ps1 -Output "%USERPROFILE%\Desktop\EmailExports" -Pick
echo.
echo Export complete!
pause
