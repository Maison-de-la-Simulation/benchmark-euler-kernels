find_package(GTest REQUIRED)

add_executable(cosine_advection_test)
target_link_libraries(cosine_advection_test PRIVATE euler_operators GTest::gtest Kokkos::kokkos)
target_sources(cosine_advection_test PRIVATE main.cpp cosine_advection_test.cpp)
gtest_discover_tests(cosine_advection_test DISCOVERY_MODE PRE_TEST)

add_executable(euler_tests)
target_link_libraries(euler_tests PRIVATE euler_operators euler_utils GTest::gtest Kokkos::kokkos)
target_sources(euler_tests PRIVATE main.cpp test_prim_to_cons.cpp test_cons_to_prim.cpp)
gtest_discover_tests(euler_tests DISCOVERY_MODE PRE_TEST)
