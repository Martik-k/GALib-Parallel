cmake_minimum_required(VERSION 3.16)

project(GALib-Parallel VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(GALIB_BUILD_EXAMPLES "Build GALib example executables" ON)
option(GALIB_BUILD_PYTHON_TOOLS "Enable Python visualization helper targets" ON)
option(GALIB_ENABLE_NATIVE_OPTIMIZATIONS "Enable -march=native style optimizations for consumers" OFF)
option(GALIB_CUDA_USE_FAST_MATH "Enable CUDA --use_fast_math for ga_cuda target" ON)
set(GALIB_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures for ga_cuda (for example: native, 75, 86, or 75;86)")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
endif ()

find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(MPI QUIET)
find_package(TBB REQUIRED)

set(GALIB_PACKAGE_HAS_MPI 0)
set(GALIB_PACKAGE_HAS_CUDA 0)

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
    add_library(ga_island INTERFACE)
    add_library(GALib::ga_island ALIAS ga_island)
    target_link_libraries(ga_island INTERFACE
            ga_lib
            MPI::MPI_CXX
    )
    target_compile_definitions(ga_island INTERFACE GALIB_HAS_MPI=1)
    set(GALIB_PACKAGE_HAS_MPI 1)
endif ()

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)

    if (CUDAToolkit_FOUND)
        add_library(ga_cuda STATIC src/cuda/StandardGACUDA.cu)
        add_library(GALib::ga_cuda ALIAS ga_cuda)
        target_include_directories(ga_cuda PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
        target_link_libraries(ga_cuda PUBLIC ga_lib CUDA::cudart)
        target_compile_definitions(ga_cuda PUBLIC GALIB_WITH_CUDA=1)
        set_target_properties(ga_cuda PROPERTIES
                CUDA_STANDARD 20
                CUDA_STANDARD_REQUIRED ON
                CUDA_ARCHITECTURES "${GALIB_CUDA_ARCHITECTURES}"
        )
        if (GALIB_CUDA_USE_FAST_MATH)
            target_compile_options(ga_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        endif ()
        set(GALIB_PACKAGE_HAS_CUDA 1)
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

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/standard_ga_cuda_example.cpp")
        add_executable(standard_ga_cuda_example examples/standard_ga_cuda_example.cpp)
        target_link_libraries(standard_ga_cuda_example PRIVATE ga_lib)

        if (TARGET ga_cuda)
            target_link_libraries(standard_ga_cuda_example PRIVATE ga_cuda)
            target_compile_definitions(standard_ga_cuda_example PRIVATE GALIB_WITH_CUDA=1)
        endif ()
    endif ()

    if (MPI_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/island_ga_example.cpp")
        add_executable(island_example examples/island_ga_example.cpp)
        target_link_libraries(island_example PRIVATE ga_island)
    endif ()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cellular_ga_example.cpp")
        add_executable(cellular_example examples/cellular_ga_example.cpp)
        target_link_libraries(cellular_example PRIVATE ga_lib)
    endif ()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/differential_evolution_example.cpp")
        add_executable(de_example examples/differential_evolution_example.cpp)
        target_link_libraries(de_example PRIVATE ga_lib)
    endif ()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/test_ga_cuda.cpp")
        add_executable(test_ga_cuda examples/test_ga_cuda.cpp)
        target_link_libraries(test_ga_cuda PRIVATE ga_lib)
        if (TARGET ga_cuda)
            target_link_libraries(test_ga_cuda PRIVATE ga_cuda)
            target_compile_definitions(test_ga_cuda PRIVATE GALIB_WITH_CUDA=1)
        endif ()
    endif ()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/minimal_user_example.cpp")
        add_executable(minimal_example examples/minimal_user_example.cpp)
        target_link_libraries(minimal_example PRIVATE ga_lib)
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

if (TARGET ga_island)
    install(TARGETS ga_island
            EXPORT GALibIslandTargets
    )
endif ()

if (TARGET ga_cuda)
    install(TARGETS ga_cuda
            EXPORT GALibTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif ()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/GALib/examples
)

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/configs/full_config_example.yaml
        DESTINATION ${CMAKE_INSTALL_DATADIR}/GALib/configs
)

install(EXPORT GALibTargets
        FILE GALibTargets.cmake
        NAMESPACE GALib::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GALib
)

if (TARGET ga_island)
    install(EXPORT GALibIslandTargets
            FILE GALibIslandTargets.cmake
            NAMESPACE GALib::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GALib
    )
endif ()

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/GALibConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/GALibConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GALib
)

enable_testing()
add_subdirectory(tests)

find_package(Doxygen QUIET)
if (DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    message(STATUS "Doxygen found — run 'cmake --build <build_dir> --target docs' to generate docs")
else ()
    message(STATUS "Doxygen not found — install it to enable the 'docs' target")
endif ()
