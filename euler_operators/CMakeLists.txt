add_library(euler_operators INTERFACE)
target_sources(
    euler_operators
    INTERFACE
        FILE_SET HEADERS
            FILES
                cons_to_prim.hpp
                euler_arrays.hpp
                godunov.hpp
                hllc.hpp
                perfect_gas.hpp
                periodic_boundary_conditions.hpp
                prim_to_cons.hpp
                time_step.hpp
                uniform_mesh.hpp
                utils.hpp
)
target_compile_features(euler_operators INTERFACE cxx_std_20)
target_link_libraries(euler_operators INTERFACE Kokkos::kokkos)
