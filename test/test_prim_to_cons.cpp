#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>
#include <prim_to_cons.hpp>

template <
        class ElementType,
        class IndexType,
        std::size_t E0,
        std::size_t E1,
        std::size_t E2,
        class LP,
        class AP>
void init_from_value_test(
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
void init_from_state_test(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<
                Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, E0, E1, E2>, LP, AP>> const&
                prim_arrays,
        EulerPrim<ElementType> const& prim)
{
    init_from_value_test(exec_space, prim_arrays.d, prim.d);
    init_from_value_test(exec_space, prim_arrays.p, prim.p);
    init_from_value_test(exec_space, prim_arrays.ux0, prim.ux0);
    init_from_value_test(exec_space, prim_arrays.ux1, prim.ux1);
    init_from_value_test(exec_space, prim_arrays.ux2, prim.ux2);
}
// include your headers

TEST(PrimToCons, ScalarVsVectorized)
{
    using real_t = double;
    using index_t = int;

    int const n = 16; // keep small for unit test

    Kokkos::DefaultExecutionSpace exec_space;
    PerfectGas<real_t> eos(1.4);

    // --- allocate ---
    auto prims_alloc = create_prim_arrays_1d<real_t>(exec_space, n * n * n);
    auto cons_alloc_ref = create_cons_arrays_1d<real_t>(exec_space, n * n * n);
    auto cons_alloc_vec = create_cons_arrays_1d<real_t>(exec_space, n * n * n);

    auto prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);

    auto cons_ref = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc_ref, n, n, n);

    auto cons_vec = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc_vec, n, n, n);

    // --- initialize with non-trivial state ---
    EulerPrim<real_t> prim {.d = 1.0, .p = 1.0, .ux0 = 0.5, .ux1 = -0.3, .ux2 = 0.1};

    init_from_state_test(exec_space, prim_arrays, prim);
    exec_space.fence();

    // --- run both implementations ---
    prim_to_cons(exec_space, as_const(prim_arrays), cons_ref, eos);
    prim_to_cons_vec(exec_space, as_const(prim_arrays), cons_vec, eos);
    exec_space.fence();

    // --- compare ---
    auto ref_h = EulerConsArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.d),
            .e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.mx0),
            .mx0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.mx1),
            .mx1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.mx2),
            .mx2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.e)};

    auto vec_h = EulerConsArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.d),
            .e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.mx0),
            .mx0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.mx1),
            .mx1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.mx2),
            .mx2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.e)};


    double const tol = 1e-12;


    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                int idx = i + (n * (j + n * k)); // layout_left flattening

                ASSERT_NEAR(ref_h.d(idx), vec_h.d(idx), tol);
                ASSERT_NEAR(ref_h.mx0(idx), vec_h.mx0(idx), tol);
                ASSERT_NEAR(ref_h.mx1(idx), vec_h.mx1(idx), tol);
                ASSERT_NEAR(ref_h.mx2(idx), vec_h.mx2(idx), tol);
                ASSERT_NEAR(ref_h.e(idx), vec_h.e(idx), tol);
            }
}
