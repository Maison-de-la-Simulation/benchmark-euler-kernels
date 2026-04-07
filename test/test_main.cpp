#include <cmath>
#include <iostream>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>


int main(int argc, char** argv)
{
    int ret = -1;
    Kokkos::initialize(argc, argv);
    {
        Kokkos::print_configuration(std::cout);
        ::testing::InitGoogleTest(&argc, argv);
        ret = RUN_ALL_TESTS();
    }
    Kokkos::finalize();
    return ret;
}
