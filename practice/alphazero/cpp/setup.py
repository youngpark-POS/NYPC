import pybind11
from setuptools import setup, Extension
import sys
import os
import platform

# 크로스 플랫폼 컴파일러 설정
def detect_compiler():
    """컴파일러 자동 감지"""
    cc = os.environ.get("CC", "").lower()
    cxx = os.environ.get("CXX", "").lower()
    
    if "mingw" in cc or "mingw" in cxx or "gcc" in cc or "g++" in cxx:
        return "mingw"
    elif platform.system() == "Windows":
        return "msvc"  
    else:
        return "gcc"

compiler_type = detect_compiler()

if compiler_type == "msvc":
    print("Using MSVC for C++ compilation")
    extra_compile_args = ['/std:c++17', '/O2', '/EHsc']
    extra_link_args = []
    
    # MSVC 사용시에만 환경변수 제거
    if "CC" in os.environ:
        del os.environ["CC"]
    if "CXX" in os.environ:
        del os.environ["CXX"]
        
elif compiler_type == "mingw":
    print("Using MinGW for C++ compilation")
    extra_compile_args = ['-std=c++17', '-O3', '-fPIC']
    extra_link_args = []
    # MinGW 환경변수 보존
    
else:
    print(f"Using GCC/Clang for {platform.system()}")
    extra_compile_args = ['-std=c++17', '-O3', '-fPIC']
    extra_link_args = []

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