cmake_minimum_required(VERSION 3.10)

project(GALib-Parallel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(MPI REQUIRED)
find_package(TBB REQUIRED)

# Interface Library
add_library(ga_lib INTERFACE)
target_include_directories(ga_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ga_lib INTERFACE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
    yaml-cpp
    TBB::tbb
)

# Optional CUDA Support
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

# Executables
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

# Visualization
find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    # Target to visualize using the main visualization.py script
    add_custom_target(visualize
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/visualization.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Launching Python visualization..."
    )
endif()

enable_testing()
