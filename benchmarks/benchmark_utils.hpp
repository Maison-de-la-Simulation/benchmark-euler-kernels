#include <stdexcept>
#include <utility>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include "euler_arrays.hpp"

template <class R, class T>
R int_cast(T t)
{
    if (std::in_range<R>(t)) {
        return static_cast<R>(t);
    }
    throw std::runtime_error("Conversion cannot preserve value representation");
}

template <
        class ElementType,
        class IndexType,
        std::size_t E0,
        std::size_t E1,
        std::size_t E2,
        class LP,
        class AP>
void init_from_value(
        Kokkos::DefaultExecutionSpace const& exec_space,
        Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, E0, E1, E2>, LP, AP> const& array,
        ElementType const& value)
{
    Kokkos::parallel_for(
            "init_from_value",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {0, 0, 0},
                    {array.extent(0), array.extent(1), array.extent(2)}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                array(i, j, k) = value;
            });
}

template <
        class ElementType,
        class IndexType,
        std::size_t E0,
        std::size_t E1,
        std::size_t E2,
        class LP,
        class AP>
void init_from_state(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<
                Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, E0, E1, E2>, LP, AP>> const&
                prim_arrays,
        EulerPrim<ElementType> const& prim)
{
    init_from_value(exec_space, prim_arrays.d, prim.d);
    init_from_value(exec_space, prim_arrays.p, prim.p);
    init_from_value(exec_space, prim_arrays.ux0, prim.ux0);
    init_from_value(exec_space, prim_arrays.ux1, prim.ux1);
    init_from_value(exec_space, prim_arrays.ux2, prim.ux2);
}

template <
        class ElementType,
        class IndexType,
        std::size_t E0,
        std::size_t E1,
        std::size_t E2,
        class LP,
        class AP>
void init_from_state(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerConsArrays<
                Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, E0, E1, E2>, LP, AP>> const&
                cons_arrays,
        EulerCons<ElementType> const& cons)
{
    init_from_value(exec_space, cons_arrays.d, cons.d);
    init_from_value(exec_space, cons_arrays.e, cons.e);
    init_from_value(exec_space, cons_arrays.mx0, cons.mx0);
    init_from_value(exec_space, cons_arrays.mx1, cons.mx1);
    init_from_value(exec_space, cons_arrays.mx2, cons.mx2);
}

void set_constant_bytes_processed(benchmark::State& state, std::size_t bytes);

void set_constant_cells_processed(benchmark::State& state, std::size_t cells);
