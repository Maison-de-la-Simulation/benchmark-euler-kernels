find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(euler_simulation euler_simulation.cpp save_npy.cpp save_vtkhdf.cpp)
target_link_libraries(euler_simulation PRIVATE euler_operators euler_utils hdf5::hdf5)
