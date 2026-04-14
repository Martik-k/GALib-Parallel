cmake_minimum_required(VERSION 3.10)

project(GALib-Parallel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
endif()

find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(MPI REQUIRED)
find_package(TBB REQUIRED)

add_library(ga_lib INTERFACE)
target_include_directories(ga_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ga_lib INTERFACE -O3 -march=native -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(ga_lib INTERFACE /O2 /arch:AVX2 /fp:fast)
endif()

target_link_libraries(ga_lib INTERFACE
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        yaml-cpp
        TBB::tbb
)

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

    target_link_libraries(ga_example PRIVATE ga_lib)

    if(TARGET ga_cuda)
        target_link_libraries(ga_example PRIVATE ga_cuda)
        target_compile_definitions(ga_example PRIVATE GALIB_WITH_CUDA=1)
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/island_example.cpp")
    add_executable(island_example examples/island_example.cpp)
    target_link_libraries(island_example PRIVATE ga_lib)
endif()

find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    add_custom_target(visualize
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/visualization.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Launching Python visualization..."
    )

    add_custom_target(visualize_landscape
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/visualize_landscape.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Launching Python visualization for landscape..."
    )
endif()

enable_testing()