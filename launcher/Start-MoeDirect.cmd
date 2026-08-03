@echo off
rem ===========================================================================
rem  MoE-Direct - double-click entry point.
rem
rem  This file only starts Start-MoeDirect.ps1 from THIS folder. The bypass below
rem  is scoped to that one PowerShell process: no machine-wide or user-wide
rem  execution policy is changed, and nothing is installed.
rem  Everything the launcher does is readable in Start-MoeDirect.ps1.
rem ===========================================================================
setlocal
set "MOE_BUNDLE=%~dp0"
set "MOE_LOGS=%LOCALAPPDATA%\MoE-Direct\logs"
set "MOE_ISSUES=https://github.com/tmxkzm1925-max/moe-direct/issues"

if not exist "%MOE_BUNDLE%Start-MoeDirect.ps1" (
  echo.
  echo  Start-MoeDirect.ps1 was not found next to this file.
  echo  Extract the whole moe-direct zip into one folder, then run this file
  echo  from that folder. The launcher did not start, so there is no status line.
  echo.
  pause
  endlocal & exit /b 1
)

powershell.exe -NoProfile -NoLogo -ExecutionPolicy Bypass -File "%MOE_BUNDLE%Start-MoeDirect.ps1" %*
set "MOE_EXIT=%ERRORLEVEL%"

if "%MOE_EXIT%"=="0" goto :quit
if "%MOE_EXIT%"=="2" (
  echo.
  echo  Cancelled. Nothing was started, and no file was deleted.
  goto :quit
)

echo.
echo  ---------------------------------------------------------------------
echo   MoE-Direct stopped. Exit code %MOE_EXIT%.
echo   The reason is the last line above, "[moe-launcher] status=...".
echo.
echo   Logs:   %MOE_LOGS%
echo   Report: %MOE_ISSUES%
echo           Attach the newest launcher_*.jsonl from the Logs folder.
echo  ---------------------------------------------------------------------
echo.
pause

:quit
endlocal & exit /b %MOE_EXIT%
