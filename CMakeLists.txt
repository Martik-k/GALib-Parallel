cmake_minimum_required(VERSION 3.10)

project(GALib-Parallel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(GALIB_BUILD_EXAMPLES "Build GALib example executables" ON)
option(GALIB_BUILD_PYTHON_TOOLS "Enable Python visualization helper targets" ON)
option(GALIB_ENABLE_NATIVE_OPTIMIZATIONS "Enable -march=native style optimizations for consumers" OFF)
option(GALIB_CUDA_USE_FAST_MATH "Enable CUDA --use_fast_math for ga_cuda target" ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
endif ()

find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(MPI QUIET)
find_package(TBB REQUIRED)

add_library(ga_lib INTERFACE)
add_library(GALib::ga_lib ALIAS ga_lib)
target_include_directories(ga_lib INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if (GALIB_ENABLE_NATIVE_OPTIMIZATIONS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ga_lib INTERFACE -O3 -march=native -ffast-math)
elseif (GALIB_ENABLE_NATIVE_OPTIMIZATIONS AND CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(ga_lib INTERFACE /O2 /arch:AVX2 /fp:fast)
endif ()

target_link_libraries(ga_lib INTERFACE
        OpenMP::OpenMP_CXX
        yaml-cpp
        TBB::tbb
)

if (MPI_FOUND)
    target_link_libraries(ga_lib INTERFACE MPI::MPI_CXX)
    target_compile_definitions(ga_lib INTERFACE GALIB_HAS_MPI=1)
endif ()

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)

    if (CUDAToolkit_FOUND)
        add_library(ga_cuda STATIC src/cuda/StandardGACUDA.cu)
        target_include_directories(ga_cuda PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(ga_cuda PUBLIC CUDA::cudart)
        set_target_properties(ga_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
        if (GALIB_CUDA_USE_FAST_MATH)
            target_compile_options(ga_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        endif ()
        message(STATUS "CUDA toolkit found, ga_cuda backend enabled")
    endif ()
endif ()

if (GALIB_BUILD_EXAMPLES)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/standard_ga_example.cpp")
        add_executable(ga_example examples/standard_ga_example.cpp)
        target_link_libraries(ga_example PRIVATE ga_lib)

        if (TARGET ga_cuda)
            target_link_libraries(ga_example PRIVATE ga_cuda)
            target_compile_definitions(ga_example PRIVATE GALIB_WITH_CUDA=1)
        endif ()
    endif ()

    if (MPI_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/island_ga_example.cpp")
        add_executable(island_example examples/island_ga_example.cpp)
        target_link_libraries(island_example PRIVATE ga_lib)
    endif ()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cellular_ga_example.cpp")
        add_executable(cellular_example examples/cellular_ga_example.cpp)
        target_link_libraries(cellular_example PRIVATE ga_lib)
    endif ()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/differential_evolution_example.cpp")
        add_executable(de_example examples/differential_evolution_example.cpp)
        target_link_libraries(de_example PRIVATE ga_lib)
    endif ()

    if (GALIB_BUILD_PYTHON_TOOLS)
        find_package(Python3 COMPONENTS Interpreter)

        if (Python3_Interpreter_FOUND)
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
        endif ()
    endif ()
endif ()

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GALibConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/GALibConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GALib
)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/GALibConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

install(TARGETS ga_lib
        EXPORT GALibTargets
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT GALibTargets
        FILE GALibTargets.cmake
        NAMESPACE GALib::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GALib
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GALibConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/GALibConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GALib
)

enable_testing()
