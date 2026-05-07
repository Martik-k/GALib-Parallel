include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_standard_ga_cuda
    test_standard_ga_cuda.cpp
)

target_compile_features(test_standard_ga_cuda PRIVATE cxx_std_20)

target_link_libraries(test_standard_ga_cuda
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        ga_lib
)

if (TARGET ga_cuda)
    target_link_libraries(test_standard_ga_cuda PRIVATE ga_cuda)
    target_compile_definitions(test_standard_ga_cuda PRIVATE GALIB_WITH_CUDA=1)
endif()

include(GoogleTest)
gtest_discover_tests(test_standard_ga_cuda
    PROPERTIES TIMEOUT 120
)
