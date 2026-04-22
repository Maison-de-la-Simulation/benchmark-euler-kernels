find_package(GTest REQUIRED)

add_executable(cosine_advection_test main.cpp cosine_advection_test.cpp)
target_link_libraries(cosine_advection_test PRIVATE euler_operators GTest::gtest)

add_executable(euler_tests)
target_link_libraries(
    euler_tests
    PRIVATE GTest::gtest_main euler_operators Kokkos::kokkos euler_utils
)
target_sources(euler_tests PRIVATE main.cpp test_prim_to_cons.cpp test_cons_to_prim.cpp test_godunov.cpp test_time_step.cpp)

gtest_discover_tests(cosine_advection_test DISCOVERY_MODE PRE_TEST)
