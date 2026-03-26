
find_package(GTest REQUIRED)

add_executable(euler_tests)
target_link_libraries(
    euler_tests
    PRIVATE GTest::gtest_main euler_operators Kokkos::kokkos
)
target_sources(
    euler_tests
    PRIVATE
        test_main.cpp
        test_prim_to_cons.cpp
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(euler_tests)
