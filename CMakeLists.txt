cmake_minimum_required(VERSION 3.10)

project(GALib-Parallel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(MPI REQUIRED)

add_library(ga_lib INTERFACE)
target_include_directories(ga_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ga_lib INTERFACE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
    yaml-cpp
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/main.cpp")
    add_executable(ga_example examples/main.cpp)
    target_link_libraries(ga_example PRIVATE ga_lib)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/island_example.cpp")
    add_executable(island_example examples/island_example.cpp)
    target_link_libraries(island_example PRIVATE ga_lib)
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
