#pragma once

#include <cstddef>
#include <type_traits>

#include <Kokkos_Core.hpp>

template <class T>
struct EulerPrim
{
    T d;
    T p;
    T ux0;
    T ux1;
    T ux2;
};

template <class T>
struct EulerCons
{
    T d;
    T e;
    T mx0;
    T mx1;
    T mx2;
};

template <class T>
struct EulerFlux
{
    T d;
    T e;
    T mx0;
    T mx1;
    T mx2;
};

template <class T>
KOKKOS_FUNCTION T kinetic_energy(EulerPrim<T> const& prim) noexcept
{
    return prim.d * ((prim.ux0 * prim.ux0) + (prim.ux1 * prim.ux1) + (prim.ux2 * prim.ux2)) / 2;
}

template <class T>
KOKKOS_FUNCTION T kinetic_energy(EulerCons<T> const& cons) noexcept
{
    return ((cons.mx0 * cons.mx0) + (cons.mx1 * cons.mx1) + (cons.mx2 * cons.mx2)) / cons.d / 2;
}

template <class T>
KOKKOS_FUNCTION constexpr T internal_energy(EulerCons<T> const& cons) noexcept
{
    return cons.e - kinetic_energy(cons);
}

template <class T>
KOKKOS_FUNCTION EulerCons<T> to_cons(EulerPrim<T> const& prim, T const int_e) noexcept
{
    return {.d = prim.d,
            .e = kinetic_energy(prim) + int_e,
            .mx0 = prim.d * prim.ux0,
            .mx1 = prim.d * prim.ux1,
            .mx2 = prim.d * prim.ux2};
}

template <class T>
KOKKOS_FUNCTION EulerPrim<T> to_prim(EulerCons<T> const& cons, T const p) noexcept
{
    T const vol_spe = 1 / cons.d;
    return {.d = cons.d,
            .p = p,
            .ux0 = cons.mx0 * vol_spe,
            .ux1 = cons.mx1 * vol_spe,
            .ux2 = cons.mx2 * vol_spe};
}

template <class View>
struct EulerPrimArrays
{
    View d;
    View p;
    View ux0;
    View ux1;
    View ux2;
};

template <class MdspanOut, class ElementType, class Extents, class LP, class AP>
EulerPrimArrays<MdspanOut> as(
        EulerPrimArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& prim_arrays) noexcept
{
    return {.d = prim_arrays.d,
            .p = prim_arrays.p,
            .ux0 = prim_arrays.ux0,
            .ux1 = prim_arrays.ux1,
            .ux2 = prim_arrays.ux2};
}

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2, class LP>
EulerPrimArrays<Kokkos::mdspan<T const, Kokkos::extents<IndexType, E0, E1, E2>, LP>> as_const(
        EulerPrimArrays<Kokkos::mdspan<T, Kokkos::extents<IndexType, E0, E1, E2>, LP>> const&
                prim_arrays) noexcept
{
    return as<Kokkos::mdspan<T const, Kokkos::extents<IndexType, E0, E1, E2>, LP>>(prim_arrays);
}

template <class ElementType, class Extents, class LP, class AP, class... Args>
auto subspan(
        EulerPrimArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& prim_arrays,
        Args const&... args) noexcept
{
    return EulerPrimArrays {
            .d = Kokkos::submdspan(prim_arrays.d, args...),
            .p = Kokkos::submdspan(prim_arrays.p, args...),
            .ux0 = Kokkos::submdspan(prim_arrays.ux0, args...),
            .ux1 = Kokkos::submdspan(prim_arrays.ux1, args...),
            .ux2 = Kokkos::submdspan(prim_arrays.ux2, args...)};
}

template <class ElementType, class Extents, class LP, class AP>
EulerPrimArrays<typename AP::data_handle_type> data_handle(
        EulerPrimArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& prim_arrays) noexcept
{
    return {.d = prim_arrays.d.data_handle(),
            .p = prim_arrays.p.data_handle(),
            .ux0 = prim_arrays.ux0.data_handle(),
            .ux1 = prim_arrays.ux1.data_handle(),
            .ux2 = prim_arrays.ux2.data_handle()};
}

template <class MdspanOut, class View>
EulerPrimArrays<MdspanOut> to_mdspan(
        EulerPrimArrays<View> const& prim_arrays,
        typename MdspanOut::index_type nx,
        typename MdspanOut::index_type ny,
        typename MdspanOut::index_type nz) noexcept
{
    return {.d = MdspanOut(prim_arrays.d.data(), nx, ny, nz),
            .p = MdspanOut(prim_arrays.p.data(), nx, ny, nz),
            .ux0 = MdspanOut(prim_arrays.ux0.data(), nx, ny, nz),
            .ux1 = MdspanOut(prim_arrays.ux1.data(), nx, ny, nz),
            .ux2 = MdspanOut(prim_arrays.ux2.data(), nx, ny, nz)};
}

template <
        class ElementType,
        class IndexType,
        std::size_t E0,
        std::size_t E1,
        std::size_t E2,
        class LP,
        class AP>
KOKKOS_FUNCTION EulerPrim<std::remove_const_t<ElementType>> load(
        EulerPrimArrays<
                Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, E0, E1, E2>, LP, AP>> const&
                prim_arrays,
        IndexType const i,
        IndexType const j,
        IndexType const k) noexcept
{
    return {.d = prim_arrays.d(i, j, k),
            .p = prim_arrays.p(i, j, k),
            .ux0 = prim_arrays.ux0(i, j, k),
            .ux1 = prim_arrays.ux1(i, j, k),
            .ux2 = prim_arrays.ux2(i, j, k)};
}

template <class T, class IndexType>
KOKKOS_FUNCTION EulerPrim<std::remove_const_t<T>> load(
        EulerPrimArrays<T*> const& prim_ptrs,
        IndexType const i) noexcept
{
    return {.d = prim_ptrs.d[i],
            .p = prim_ptrs.p[i],
            .ux0 = prim_ptrs.ux0[i],
            .ux1 = prim_ptrs.ux1[i],
            .ux2 = prim_ptrs.ux2[i]};
}

template <
        class ElementType,
        class IndexType,
        std::size_t E0,
        std::size_t E1,
        std::size_t E2,
        class LP,
        class AP>
KOKKOS_FUNCTION void store(
        EulerPrim<ElementType> const& prim,
        EulerPrimArrays<
                Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, E0, E1, E2>, LP, AP>> const&
                prim_arrays,
        IndexType const i,
        IndexType const j,
        IndexType const k) noexcept
{
    prim_arrays.d(i, j, k) = prim.d;
    prim_arrays.p(i, j, k) = prim.p;
    prim_arrays.ux0(i, j, k) = prim.ux0;
    prim_arrays.ux1(i, j, k) = prim.ux1;
    prim_arrays.ux2(i, j, k) = prim.ux2;
}

template <class T, class IndexType>
KOKKOS_FUNCTION void store(
        EulerPrim<T> const& prim,
        EulerPrimArrays<T*> const& prim_ptrs,
        IndexType const i) noexcept
{
    prim_ptrs.d[i] = prim.d;
    prim_ptrs.p[i] = prim.p;
    prim_ptrs.ux0[i] = prim.ux0;
    prim_ptrs.ux1[i] = prim.ux1;
    prim_ptrs.ux2[i] = prim.ux2;
}

template <class T>
EulerPrimArrays<Kokkos::View<T*>> create_prim_arrays_1d(
        Kokkos::DefaultExecutionSpace const& exec_space,
        std::size_t const n)
{
    return {.d = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "density"), n),
            .p = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "pressure"), n),
            .ux0 = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "ux0"), n),
            .ux1 = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "ux1"), n),
            .ux2 = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "ux2"), n)};
}

template <class View>
struct EulerConsArrays
{
    View d;
    View e;
    View mx0;
    View mx1;
    View mx2;
};

template <class MdspanOut, class ElementType, class Extents, class LP, class AP>
EulerConsArrays<MdspanOut> as(
        EulerConsArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& cons_arrays) noexcept
{
    return {.d = cons_arrays.d,
            .e = cons_arrays.e,
            .mx0 = cons_arrays.mx0,
            .mx1 = cons_arrays.mx1,
            .mx2 = cons_arrays.mx2};
}

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2, class LP>
EulerConsArrays<Kokkos::mdspan<T const, Kokkos::extents<IndexType, E0, E1, E2>, LP>> as_const(
        EulerConsArrays<Kokkos::mdspan<T, Kokkos::extents<IndexType, E0, E1, E2>, LP>> const&
                cons_arrays) noexcept
{
    return as<Kokkos::mdspan<T const, Kokkos::extents<IndexType, E0, E1, E2>, LP>>(cons_arrays);
}

template <class ElementType, class Extents, class LP, class AP, class... Args>
auto subspan(
        EulerConsArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& cons_arrays,
        Args const&... args) noexcept
{
    return EulerConsArrays {
            .d = Kokkos::submdspan(cons_arrays.d, args...),
            .e = Kokkos::submdspan(cons_arrays.e, args...),
            .mx0 = Kokkos::submdspan(cons_arrays.mx0, args...),
            .mx1 = Kokkos::submdspan(cons_arrays.mx1, args...),
            .mx2 = Kokkos::submdspan(cons_arrays.mx2, args...)};
}

template <class ElementType, class Extents, class LP, class AP>
EulerConsArrays<typename AP::data_handle_type> data_handle(
        EulerConsArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& cons_arrays) noexcept
{
    return {.d = cons_arrays.d.data_handle(),
            .e = cons_arrays.e.data_handle(),
            .mx0 = cons_arrays.mx0.data_handle(),
            .mx1 = cons_arrays.mx1.data_handle(),
            .mx2 = cons_arrays.mx2.data_handle()};
}

template <class MdspanOut, class View>
EulerConsArrays<MdspanOut> to_mdspan(
        EulerConsArrays<View> const& prim_arrays,
        typename MdspanOut::index_type nx,
        typename MdspanOut::index_type ny,
        typename MdspanOut::index_type nz) noexcept
{
    return {.d = MdspanOut(prim_arrays.d.data(), nx, ny, nz),
            .e = MdspanOut(prim_arrays.e.data(), nx, ny, nz),
            .mx0 = MdspanOut(prim_arrays.mx0.data(), nx, ny, nz),
            .mx1 = MdspanOut(prim_arrays.mx1.data(), nx, ny, nz),
            .mx2 = MdspanOut(prim_arrays.mx2.data(), nx, ny, nz)};
}

template <
        class ElementType,
        class IndexType,
        std::size_t E0,
        std::size_t E1,
        std::size_t E2,
        class LP,
        class AP>
KOKKOS_FUNCTION EulerCons<std::remove_const_t<ElementType>> load(
        EulerConsArrays<
                Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, E0, E1, E2>, LP, AP>> const&
                cons_arrays,
        IndexType const i,
        IndexType const j,
        IndexType const k) noexcept
{
    return {.d = cons_arrays.d(i, j, k),
            .e = cons_arrays.e(i, j, k),
            .mx0 = cons_arrays.mx0(i, j, k),
            .mx1 = cons_arrays.mx1(i, j, k),
            .mx2 = cons_arrays.mx2(i, j, k)};
}

template <class T, class IndexType>
KOKKOS_FUNCTION EulerCons<std::remove_const_t<T>> load(
        EulerConsArrays<T*> const& cons_ptrs,
        IndexType const i) noexcept
{
    return {.d = cons_ptrs.d[i],
            .e = cons_ptrs.e[i],
            .mx0 = cons_ptrs.mx0[i],
            .mx1 = cons_ptrs.mx1[i],
            .mx2 = cons_ptrs.mx2[i]};
}

template <
        class ElementType,
        class IndexType,
        std::size_t E0,
        std::size_t E1,
        std::size_t E2,
        class LP,
        class AP>
KOKKOS_FUNCTION void store(
        EulerCons<ElementType> const& cons,
        EulerConsArrays<
                Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, E0, E1, E2>, LP, AP>> const&
                cons_arrays,
        IndexType const i,
        IndexType const j,
        IndexType const k) noexcept
{
    cons_arrays.d(i, j, k) = cons.d;
    cons_arrays.e(i, j, k) = cons.e;
    cons_arrays.mx0(i, j, k) = cons.mx0;
    cons_arrays.mx1(i, j, k) = cons.mx1;
    cons_arrays.mx2(i, j, k) = cons.mx2;
}

template <class T, class IndexType>
KOKKOS_FUNCTION void store(
        EulerCons<T> const& cons,
        EulerConsArrays<T*> const& cons_ptrs,
        IndexType const i) noexcept
{
    cons_ptrs.d[i] = cons.d;
    cons_ptrs.e[i] = cons.e;
    cons_ptrs.mx0[i] = cons.mx0;
    cons_ptrs.mx1[i] = cons.mx1;
    cons_ptrs.mx2[i] = cons.mx2;
}

template <class ElementType, class Extents, class LP, class AP>
std::size_t size(
        EulerPrimArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& prim_arrays) noexcept
{
    return prim_arrays.d.size();
}

template <class ElementType, class Extents, class LP, class AP>
std::size_t size(
        EulerConsArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& cons_arrays) noexcept
{
    return cons_arrays.d.size();
}

template <class ElementType, class Extents, class LP, class AP>
std::size_t size_bytes(
        EulerPrimArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& prim_arrays) noexcept
{
    std::size_t const nb_arrays = 5;
    std::size_t const data_type_size_bytes = sizeof(ElementType);
    std::size_t const elements_per_array = size(prim_arrays);
    return nb_arrays * data_type_size_bytes * elements_per_array;
}

template <class ElementType, class Extents, class LP, class AP>
std::size_t size_bytes(
        EulerConsArrays<Kokkos::mdspan<ElementType, Extents, LP, AP>> const& cons_arrays) noexcept
{
    std::size_t const nb_arrays = 5;
    std::size_t const data_type_size_bytes = sizeof(ElementType);
    std::size_t const elements_per_array = size(cons_arrays);
    return nb_arrays * data_type_size_bytes * elements_per_array;
}

template <class T>
EulerConsArrays<Kokkos::View<T*>> create_cons_arrays_1d(
        Kokkos::DefaultExecutionSpace const& exec_space,
        std::size_t const n)
{
    return {.d = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "density"), n),
            .e = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "energy"), n),
            .mx0 = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "mx0"), n),
            .mx1 = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "mx1"), n),
            .mx2 = Kokkos::View<T*>(Kokkos::view_alloc(exec_space, "mx2"), n)};
}
