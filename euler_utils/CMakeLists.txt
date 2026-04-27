add_library(euler_utils INTERFACE)
target_sources(euler_utils INTERFACE FILE_SET HEADERS FILES utils.hpp)
target_compile_features(euler_utils INTERFACE cxx_std_20)
target_link_libraries(euler_utils INTERFACE euler_operators Kokkos::kokkos)
