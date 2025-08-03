# NYPC AlphaZero C++ GameBoard 빌드 가이드

## 개요

NYPC AlphaZero 프로젝트의 C++ GameBoard 모듈을 빌드하는 방법을 설명합니다. 이 모듈은 Python에서 사용할 수 있는 고성능 C++ 확장 모듈입니다.

## 시스템 요구사항

### 필수 소프트웨어

- **Python 3.7+** (권장: 3.11)
- **CMake 3.12+**
- **Visual Studio 2017+** (MSVC 컴파일러)
- **pybind11** (`pip install pybind11`)

### 권장 환경

- Windows 10/11
- Visual Studio 2022 (Community/Professional)
- Conda 또는 venv 가상환경

## 빌드 방법

### 방법 1: 자동 빌드 스크립트 (권장)

가장 간단한 방법입니다:

```batch
# cpp 디렉토리로 이동
cd practice/alphazero/cpp

# 빌드 스크립트 실행
build.bat
```

스크립트가 자동으로:
- 환경 검사 (Python, CMake, MSVC)
- Python 버전 자동 감지 (3.7-3.12 지원)
- 이전 빌드 정리
- CMake 설정 및 빌드
- 빌드된 모듈을 메인 디렉토리로 자동 이동
- 모듈 테스트

### 방법 2: 수동 빌드

세부 제어가 필요한 경우:

```batch
# 1. 빌드 디렉토리 준비
rmdir /s /q build  # 기존 빌드 정리
mkdir build
cd build

# 2. CMake 설정 (Python 경로를 본인 환경에 맞게 수정)
cmake -DPython3_EXECUTABLE="C:/Users/[사용자명]/miniconda3/envs/[환경명]/python.exe" ..

# 3. 빌드 실행
cmake --build . --config Release

# 4. 모듈 복사
cd ..
copy build\Release\fast_game_board.cp*-win_amd64.pyd fast_game_board.pyd
```

## 빌드 확인

빌드가 완료되면 다음 명령으로 테스트하세요:

```python
import fast_game_board

# 테스트 보드 생성
board_data = [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2] for _ in range(10)]
board = fast_game_board.GameBoard(board_data)

print(f"현재 플레이어: {board.get_current_player()}")
print(f"액션 공간 크기: {board.get_action_space_size()}")
print(f"게임 종료: {board.is_game_over()}")
```

## 문제 해결

### 일반적인 오류와 해결방법

#### 1. "Python이 설치되지 않았거나 PATH에 없습니다"
- Python이 시스템 PATH에 추가되어 있는지 확인
- `python --version` 명령이 작동하는지 확인

#### 2. "CMake가 설치되지 않았거나 PATH에 없습니다"
- CMake 설치: https://cmake.org/download/
- Visual Studio Installer에서 "CMake tools for Visual Studio" 설치

#### 3. "MSVC 컴파일러를 찾을 수 없습니다"
- Visual Studio 설치시 "Desktop development with C++" 워크로드 선택
- Developer Command Prompt for VS에서 실행
- 또는 Visual Studio Code에서 "C/C++ Extension Pack" 설치

#### 4. "pybind11이 설치되지 않았습니다"
```bash
pip install pybind11
```

#### 5. "DLL load failed while importing"
- **해결책**: `build.bat` 스크립트 사용 (자동 버전 매칭)
- Python 버전과 빌드된 모듈의 버전 불일치
- 수동 빌드시: Python 3.11용으로 빌드했다면 Python 3.11에서 실행
- Conda 환경이 활성화되어 있는지 확인

#### 6. Segmentation Fault 또는 Access Violation
- 입력 데이터 크기 확인 (10x17 배열)
- 빌드를 Release 모드로 다시 시도
- Visual Studio에서 디버그 모드로 빌드하여 문제 위치 파악

### Python 환경 설정

#### Conda 환경 사용시
```bash
# 새 환경 생성
conda create -n nypc_dev python=3.11
conda activate nypc_dev

# 필요한 패키지 설치
pip install pybind11 numpy
```

#### venv 사용시
```bash
# 가상환경 생성
python -m venv nypc_env
nypc_env\Scripts\activate

# 필요한 패키지 설치
pip install pybind11 numpy
```

## 빌드 시스템 구조

```
practice/alphazero/cpp/
├── src/
│   ├── game_board.h          # GameBoard 클래스 헤더
│   ├── game_board.cpp        # GameBoard 구현
│   └── python_binding.cpp    # Python 바인딩
├── CMakeLists.txt            # CMake 설정
├── setup.py                  # setuptools 빌드 (대안)
├── build.bat                 # Windows 자동 빌드 스크립트
├── BUILD.md                  # 이 문서
└── build/                    # 빌드 결과물 (자동 생성)
```

## 성능 정보

- **GameBoard 크기**: 10x17 (170 셀)
- **액션 공간**: ~8,246개 액션
- **메모리 사용량**: ~50MB (액션 매핑 테이블 포함)
- **성능 향상**: Python 대비 약 10-100배 빠름

## 주요 API

### GameBoard 클래스

```cpp
// 생성자
GameBoard(const std::vector<std::vector<int>>& initial_board);

// 게임 상태
int get_current_player() const;
bool is_game_over() const;
int get_winner() const;

// 액션 관련
std::vector<Move> get_valid_moves() const;
bool make_move(int r1, int c1, int r2, int c2, int player);
int get_action_space_size() const;
int encode_move(int r1, int c1, int r2, int c2) const;
Move decode_action(int action_idx) const;

// 상태 정보
const std::vector<std::vector<int>>& get_board() const;
std::pair<int, int> get_score() const;
float get_reward(int player) const;
```

## 기여하기

1. 코드 수정 후 빌드 테스트
2. 새로운 기능 추가시 단위 테스트 작성
3. 성능 최적화시 벤치마크 결과 포함
4. 빌드 시스템 수정시 이 문서 업데이트

## 라이선스

NYPC 프로젝트 라이선스를 따릅니다.