#pragma once

#include <Kokkos_Core.hpp>

void write_imagedata(
        char const* filename,
        char const* dataset_name,
        Kokkos::mdspan<double const, Kokkos::dextents<int, 3>, Kokkos::layout_left> const& mds);
