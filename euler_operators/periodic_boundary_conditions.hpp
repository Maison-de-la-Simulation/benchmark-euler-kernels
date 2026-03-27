#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void boundary_conditions_periodic(
        Kokkos::DefaultExecutionSpace const& exec_space,
        Kokkos::mdspan<T, Kokkos::extents<IndexType, E0, E1, E2>, Kokkos::layout_left> const& mds,
        int const gw)
{
    using Kokkos::ALL;
    Kokkos::View<T***, Kokkos::LayoutLeft> const
            view(mds.data_handle(), mds.extent(0), mds.extent(1), mds.extent(2));
    Kokkos::deep_copy(
            exec_space,
            Kokkos::subview(view, Kokkos::make_pair(0, gw), ALL, ALL),
            Kokkos::
                    subview(view,
                            Kokkos::make_pair(view.extent(0) - (2 * gw), view.extent(0) - gw),
                            ALL,
                            ALL));
    Kokkos::deep_copy(
            exec_space,
            Kokkos::subview(view, Kokkos::make_pair(view.extent(0) - gw, view.extent(0)), ALL, ALL),
            Kokkos::subview(view, Kokkos::make_pair(gw, 2 * gw), ALL, ALL));

    Kokkos::deep_copy(
            exec_space,
            Kokkos::subview(view, ALL, Kokkos::make_pair(0, gw), ALL),
            Kokkos::
                    subview(view,
                            ALL,
                            Kokkos::make_pair(view.extent(1) - (2 * gw), view.extent(1) - gw),
                            ALL));
    Kokkos::deep_copy(
            exec_space,
            Kokkos::subview(view, ALL, Kokkos::make_pair(view.extent(1) - gw, view.extent(1)), ALL),
            Kokkos::subview(view, ALL, Kokkos::make_pair(gw, 2 * gw), ALL));

    Kokkos::deep_copy(
            exec_space,
            Kokkos::subview(view, ALL, ALL, Kokkos::make_pair(0, gw)),
            Kokkos::
                    subview(view,
                            ALL,
                            ALL,
                            Kokkos::make_pair(view.extent(2) - (2 * gw), view.extent(2) - gw)));
    Kokkos::deep_copy(
            exec_space,
            Kokkos::subview(view, ALL, ALL, Kokkos::make_pair(view.extent(2) - gw, view.extent(2))),
            Kokkos::subview(view, ALL, ALL, Kokkos::make_pair(gw, 2 * gw)));
}

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void boundary_conditions_periodic(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        int const gw)
{
    boundary_conditions_periodic(exec_space, cons_arrays.d, gw);
    boundary_conditions_periodic(exec_space, cons_arrays.e, gw);
    boundary_conditions_periodic(exec_space, cons_arrays.mx0, gw);
    boundary_conditions_periodic(exec_space, cons_arrays.mx1, gw);
    boundary_conditions_periodic(exec_space, cons_arrays.mx2, gw);
}
