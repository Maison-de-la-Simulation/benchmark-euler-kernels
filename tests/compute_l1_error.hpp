#pragma once

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <uniform_mesh.hpp>

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
T compute_l1_error(
        Kokkos::DefaultExecutionSpace const& exec_space,
        Kokkos::mdspan<T const, Kokkos::extents<IndexType, E0, E1, E2>, Kokkos::layout_left> const&
                array1,
        Kokkos::mdspan<T const, Kokkos::extents<IndexType, E0, E1, E2>, Kokkos::layout_left> const&
                array2)
{
    T error {};
    Kokkos::parallel_reduce(
            "compute_l1_error",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {0, 0, 0},
                    {array1.extent(0), array1.extent(1), array1.extent(2)}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k, T& error_loc) {
                T const diff = array1(i, j, k) - array2(i, j, k);
                error_loc += Kokkos::abs(diff);
            },
            Kokkos::Sum<T>(error));
    return error / array1.size();
}
