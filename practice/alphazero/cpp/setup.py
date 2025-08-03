import pybind11
from setuptools import setup, Extension
import os

# Windows conda 환경에서 MSVC 전용 설정
print("Using MSVC for C++ compilation in conda environment")

# MSVC 전용 컴파일 플래그 - Release 모드로 변경
extra_compile_args = [
    '/std:c++17',      # C++17 표준
    '/O2',             # 최적화 활성화
    '/EHsc',           # 예외 처리
    '/bigobj',         # 큰 오브젝트 파일 지원
    '/wd4996',         # 사용 중단 경고 억제
    '/DNDEBUG',        # Release 정의
]
extra_link_args = []

# Python 아키텍처 확인하고 컴파일러 아키텍처 맞추기
import platform
import distutils.util
python_arch = platform.architecture()[0]
print(f"Python architecture: {python_arch}")

# MSVC 컴파일러 환경 설정
os.environ['DISTUTILS_USE_SDK'] = '1'
os.environ['MSSdk'] = '1'

# 환경변수 정리 - MSVC와 충돌 방지
for env_var in ['CC', 'CXX']:
    if env_var in os.environ:
        del os.environ[env_var]

# PATH에서 MSYS2/Git의 링커 제거 - MSVC 링커만 사용
import sys
if sys.platform == 'win32':
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    # MSYS2, Git의 bin 디렉토리 제거
    filtered_dirs = [d for d in path_dirs if not any(
        pattern in d.lower() for pattern in [
            'git/usr/bin', 'msys2', 'mingw64/bin', 'git\\usr\\bin'
        ]
    )]
    os.environ['PATH'] = os.pathsep.join(filtered_dirs)

# C++ 확장 모듈 정의 (기존 Extension 사용)
ext_modules = [
    Extension(
        "fast_game_board",
        sources=[
            "src/game_board.cpp",
            "src/python_binding.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="fast_game_board",
    version="0.1.0",
    author="NYPC Team",
    description="Fast C++ GameBoard implementation for AlphaZero",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7",
)