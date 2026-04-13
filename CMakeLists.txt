cmake_minimum_required(VERSION 3.10)

project(GALib-Parallel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
endif()

add_library(ga_lib INTERFACE)
target_include_directories(ga_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ga_lib INTERFACE -O3 -march=native -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(ga_lib INTERFACE /O2 /arch:AVX2 /fp:fast)
endif()

find_package(OpenMP)
find_package(yaml-cpp REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        add_library(ga_cuda STATIC src/cuda/StandardGACUDA.cu)
        target_include_directories(ga_cuda PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(ga_cuda PUBLIC CUDA::cudart)
        set_target_properties(ga_cuda PROPERTIES CUDA_STANDARD 17)
        message(STATUS "CUDA toolkit found, ga_cuda backend enabled")
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/main.cpp")
    add_executable(ga_example examples/main.cpp)

    target_link_libraries(ga_example PRIVATE ga_lib yaml-cpp)

    if(TARGET ga_cuda)
        target_link_libraries(ga_example PRIVATE ga_cuda)
        target_compile_definitions(ga_example PRIVATE GALIB_WITH_CUDA=1)
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(ga_example PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP found and linked to ga_example")
    endif()
endif()

find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    add_custom_target(visualize
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/visualize_landscape.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Launching Python visualization..."
    )
endif()

enable_testing()