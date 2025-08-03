@echo off
setlocal EnableDelayedExpansion

:: NYPC AlphaZero C++ GameBoard 빌드 스크립트
echo ==========================================
echo NYPC AlphaZero C++ GameBoard Build Script
echo ==========================================
echo.

:: 환경 검사
echo [1/6] 환경 검사...

:: Python 검사
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 오류: Python이 설치되지 않았거나 PATH에 없습니다.
    echo Python 3.7 이상을 설치하고 PATH에 추가하세요.
    pause
    exit /b 1
)

:: Python 버전 출력
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   - Python: %PYTHON_VERSION%

:: pybind11 검사
python -c "import pybind11" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 오류: pybind11이 설치되지 않았습니다.
    echo 설치 명령: pip install pybind11
    pause
    exit /b 1
)

:: CMake 검사
cmake --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 오류: CMake가 설치되지 않았거나 PATH에 없습니다.
    echo CMake 3.12 이상을 설치하고 PATH에 추가하세요.
    pause
    exit /b 1
)

:: CMake 버전 출력
for /f "tokens=3" %%i in ('cmake --version 2^>^&1 ^| findstr "cmake version"') do set CMAKE_VERSION=%%i
echo   - CMake: %CMAKE_VERSION%

:: MSVC 검사
cl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 오류: MSVC 컴파일러를 찾을 수 없습니다.
    echo Visual Studio 2017 이상을 설치하거나 Developer Command Prompt에서 실행하세요.
    pause
    exit /b 1
)

echo   - MSVC: 사용 가능
echo.

:: Python 경로 찾기
echo [2/6] Python 경로 확인...
for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_PATH=%%i
echo   - Python 경로: %PYTHON_PATH%
echo.

:: 이전 빌드 정리
echo [3/6] 이전 빌드 정리...
if exist build (
    rmdir /s /q build
    echo   - 이전 빌드 디렉토리 삭제됨
)
if exist fast_game_board.pyd (
    del fast_game_board.pyd
    echo   - 이전 .pyd 파일 삭제됨
)
echo.

:: 빌드 디렉토리 생성
echo [4/6] 빌드 준비...
mkdir build
cd build
echo   - 빌드 디렉토리 생성됨
echo.

:: CMake 설정
echo [5/6] CMake 설정...
cmake -DPython3_EXECUTABLE="%PYTHON_PATH%" .. 
if %ERRORLEVEL% neq 0 (
    echo 오류: CMake 설정 실패
    cd ..
    pause
    exit /b 1
)
echo   - CMake 설정 완료
echo.

:: 빌드 실행
echo [6/6] 빌드 실행...
cmake --build . --config Release
if %ERRORLEVEL% neq 0 (
    echo 오류: 빌드 실패
    cd ..
    pause
    exit /b 1
)

:: Python 버전 감지
echo [6.1/6] Python 버전 감지...
for /f "tokens=1,2 delims=." %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do (
    set PY_MAJOR=%%i
    set PY_MINOR=%%j
)
echo   - 감지된 Python 버전: %PY_MAJOR%.%PY_MINOR%

:: 빌드된 파일을 메인 디렉토리로 이동 (버전 자동 매칭)
echo [6.2/6] 빌드된 파일 이동...
cd ..
set SOURCE_FILE=build\Release\fast_game_board.cp%PY_MAJOR%%PY_MINOR%-win_amd64.pyd
set TARGET_FILE=..\fast_game_board.pyd

if exist "%SOURCE_FILE%" (
    move "%SOURCE_FILE%" "%TARGET_FILE%" >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo 경고: .pyd 파일 이동 실패, 수동으로 이동하세요.
    ) else (
        echo   - fast_game_board.pyd를 메인 디렉토리로 이동 완료
    )
) else (
    echo 경고: 예상 파일을 찾을 수 없음: %SOURCE_FILE%
    echo 사용 가능한 파일들:
    dir build\Release\fast_game_board*.pyd 2>nul
    echo 수동으로 이동하세요: move build\Release\fast_game_board*.pyd ..\fast_game_board.pyd
)
echo.

:: 빌드 성공
echo ==========================================
echo 빌드 성공!
echo ==========================================
echo.
echo 테스트 명령:
echo   python -c "import fast_game_board; print('Import 성공!')"
echo.

:: 간단한 테스트 실행
echo 간단한 테스트 실행 중...
python -c "import fast_game_board; board = fast_game_board.GameBoard([[1]*17 for _ in range(10)]); print(f'GameBoard 생성 성공! 액션 공간: {board.get_action_space_size()}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo 경고: 모듈 import 테스트 실패
    echo 수동으로 테스트하세요: python -c "import fast_game_board"
) else (
    echo 모듈 테스트 성공!
)

echo.
pause