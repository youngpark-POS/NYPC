import pybind11
from setuptools import setup, Extension
import sys
import os

print("Using MSVC for C++ compilation")

# MSVC 컴파일러 설정
extra_compile_args = ['/std:c++17', '/O2', '/EHsc']
extra_link_args = []

# MSYS2 환경변수 제거
if "CC" in os.environ:
    del os.environ["CC"]
if "CXX" in os.environ:
    del os.environ["CXX"]

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