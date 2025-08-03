cmake_minimum_required(VERSION 3.12)
project(fast_game_board)

# C++17 표준 사용
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 릴리즈 모드 최적화
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Windows MSVC용 릴리즈 플래그
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Python 찾기 (현재 활성화된 Python 사용)
set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
if(NOT Python3_EXECUTABLE)
    find_program(Python3_EXECUTABLE NAMES python python3 
        PATHS $ENV{CONDA_PREFIX}/bin $ENV{VIRTUAL_ENV}/bin
        NO_DEFAULT_PATH)
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11 찾기
find_package(pybind11 REQUIRED)

# 소스 파일들 (setup.py와 일치)
set(SOURCES
    src/game_board.cpp
    src/python_binding.cpp
)

# Python 확장 모듈 생성 (Python3 사용)
pybind11_add_module(fast_game_board ${SOURCES})

# Python 라이브러리 명시적으로 링크
target_link_libraries(fast_game_board PRIVATE Python3::Python)

# 컴파일러별 최적화 플래그
if(MSVC)
    target_compile_options(fast_game_board PRIVATE /O2 /openmp)
else()
    target_compile_options(fast_game_board PRIVATE -O3 -march=native -fopenmp)
    target_link_libraries(fast_game_board PRIVATE gomp)
endif()

# 헤더 파일 경로
target_include_directories(fast_game_board PRIVATE src)

# Python에서 import할 수 있도록 확장자 설정
set_target_properties(fast_game_board PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)