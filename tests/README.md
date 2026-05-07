# GALib-Parallel — Test Suite

Tests use [GoogleTest](https://github.com/google/googletest), fetched automatically by CMake on the first configure — no manual installation needed.

---

## Prerequisites

| Requirement | Minimum version |
|-------------|----------------|
| CMake | 3.16 |
| C++ compiler | GCC 11 / Clang 14 (C++20) |
| CUDA toolkit | Optional — CUDA tests are skipped without it |
| Internet access | First configure only (GoogleTest download) |

---

## Build and run

```bash
# Configure (from project root)
cmake -B build

# Build only the test binary
cmake --build build --target test_standard_ga_cuda

# Or build everything
cmake --build build

# Run all tests
ctest --test-dir build --verbose

# Filter by name pattern
ctest --test-dir build -R "Sphere|Rastrigin"
ctest --test-dir build -R "FileLogging"

# Increase timeout if GPU is slow
ctest --test-dir build --timeout 180
```

> **Always build before running `ctest`.** `gtest_discover_tests` inspects the binary at configure time; if the binary does not exist yet, CTest will have nothing to run.

---

## How to add a new test file

1. Create `tests/test_<algorithm_name>.cpp` (see the structure below).
2. In `tests/CMakeLists.txt`, add a new block:

```cmake
add_executable(test_<algorithm_name>
    test_<algorithm_name>.cpp
)
target_compile_features(test_<algorithm_name> PRIVATE cxx_std_20)
target_link_libraries(test_<algorithm_name>
    PRIVATE GTest::gtest GTest::gtest_main ga_lib
)
if (TARGET ga_cuda)
    target_link_libraries(test_<algorithm_name> PRIVATE ga_cuda)
    target_compile_definitions(test_<algorithm_name> PRIVATE GALIB_WITH_CUDA=1)
endif()
gtest_discover_tests(test_<algorithm_name> PROPERTIES TIMEOUT 120)
```

3. Re-run `cmake -B build` so the new target is discovered, then build and test as above.

---

## How to add a test case

All tests live inside a **fixture class** that inherits `::testing::Test`. The fixture provides shared helpers (`makeOperators()`, `makePopulation()`) so you don't repeat boilerplate.

```cpp
// Use the existing fixture for StandardGA/CUDA tests
TEST_F(StandardGACUDATest, MyNewTest) {
    // Arrange
    SphereFunction<double> ff(10, -5.12, 5.12);
    auto ops = makeOperators();
    auto pop = makePopulation();

    // Act
    cuda::StandardGACUDA<double> ga(
        ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
        0.05, 0.8, /*max_gen=*/100, /*elitism=*/true
    );
    ga.run(pop);

    // Assert
    EXPECT_LT(pop.getBestIndividual().getFitness(), 1.0);
}
```

Common GoogleTest assertions:

| Macro | Meaning |
|-------|---------|
| `EXPECT_LT(a, b)` | a < b (non-fatal) |
| `EXPECT_LE(a, b)` | a ≤ b (non-fatal) |
| `EXPECT_EQ(a, b)` | a == b (non-fatal) |
| `ASSERT_NO_THROW(expr)` | expr does not throw (fatal on fail) |
| `ASSERT_TRUE(cond)` | cond is true (fatal on fail) |

Full reference: https://google.github.io/googletest/reference/assertions.html

---

## CUDA conditional compilation

Any test that directly uses `galib::cuda::StandardGACUDA` must be wrapped so it compiles on machines without CUDA:

```cpp
#ifdef GALIB_WITH_CUDA
TEST_F(StandardGACUDATest, MyGPUTest) {
    // ... uses cuda::StandardGACUDA ...
}
#endif
```

`GALIB_WITH_CUDA` is defined automatically by CMake when the CUDA toolkit is found. Tests outside the guard use only CPU classes and always compile.

---

## Known limitation — file/console logging on StandardGACUDA

`StandardGACUDA::run()` does **not** call `notifyLoggers()`. During evolution the population lives entirely on the GPU, and copying it back to the CPU each generation would negate the GPU throughput benefit. As a result, `enableFileLogging()` and `enableConsoleLogging()` silently have no effect when called on a `StandardGACUDA` instance.

To test logging behaviour, use `StandardGA` (the CPU variant), which shares the same `Algorithm` base class and correctly invokes loggers every generation. See `FileLoggingCreatesFile` in [test_standard_ga_cuda.cpp](test_standard_ga_cuda.cpp) for an example.
