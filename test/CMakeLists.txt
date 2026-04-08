
find_package(GTest REQUIRED)

add_executable(euler_tests)
target_link_libraries(
    euler_tests
    PRIVATE GTest::gtest_main euler_operators Kokkos::kokkos euler_utils
)
target_sources(
    euler_tests
    PRIVATE
        test_main.cpp
        test_prim_to_cons.cpp
        test_cons_to_prim.cpp
        test_time_step.cpp
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(euler_tests DISCOVERY_MODE PRE_TEST)
