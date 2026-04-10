#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <godunov.hpp>
#include <hllc.hpp>
#include <perfect_gas.hpp>
#include <uniform_mesh.hpp>

#include "utils.hpp"

namespace {

// Runs both kernels from the same initial state and returns copies of the
// resulting cons arrays so the caller can compare them field-by-field.
struct GodunovResults
{
    std::vector<double> d, e, mx0, mx1, mx2;
};

template <class MdspanType>
GodunovResults to_host(EulerConsArrays<MdspanType> const& cons_arrays)
{
    // Each field's data_handle() points into a flat 1-D device allocation.
    // We wrap it in an unmanaged View so we can use Kokkos deep_copy.
    std::size_t const n = cons_arrays.d.mapping().required_span_size();

    auto copy_field = [&](auto* ptr) {
        Kokkos::View<
                double*,
                Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                device_view(ptr, n);
        auto host_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, device_view);
        return std::vector<double>(host_mirror.data(), host_mirror.data() + n);
    };

    return GodunovResults {
            .d = copy_field(cons_arrays.d.data_handle()),
            .e = copy_field(cons_arrays.e.data_handle()),
            .mx0 = copy_field(cons_arrays.mx0.data_handle()),
            .mx1 = copy_field(cons_arrays.mx1.data_handle()),
            .mx2 = copy_field(cons_arrays.mx2.data_handle()),
    };
}
GodunovResults run_scalar(
        Kokkos::DefaultExecutionSpace& exec_space,
        int n,
        EulerPrim<double> const& prim,
        PerfectGas<double> const& eos,
        UniformMesh3d<double> const& mesh,
        double dt)
{
    auto prims_alloc = create_prim_arrays_1d<double>(exec_space, n * n * n);
    auto prim_arrays = to_mdspan<Kokkos::mdspan<
            double,
            Kokkos::dextents<int, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);
    auto cons_alloc = create_cons_arrays_1d<double>(exec_space, n * n * n);
    auto cons_arrays = to_mdspan<Kokkos::mdspan<
            double,
            Kokkos::dextents<int, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);

    init_from_state(exec_space, prim_arrays, prim);
    init_from_state(exec_space, cons_arrays, to_cons(prim, eos.internal_energy(prim.d, prim.p)));
    exec_space.fence();

    godunov(exec_space, as_const(prim_arrays), cons_arrays, eos, mesh, hllc(), dt);
    exec_space.fence();

    return to_host(cons_arrays); // assumed helper mirroring prim/cons to std::vector
}

GodunovResults run_vec(
        Kokkos::DefaultExecutionSpace& exec_space,
        int n,
        EulerPrim<double> const& prim,
        PerfectGas<double> const& eos,
        UniformMesh3d<double> const& mesh,
        double dt)
{
    auto prims_alloc = create_prim_arrays_1d<double>(exec_space, n * n * n);
    auto prim_arrays = to_mdspan<Kokkos::mdspan<
            double,
            Kokkos::dextents<int, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);
    auto cons_alloc = create_cons_arrays_1d<double>(exec_space, n * n * n);
    auto cons_arrays = to_mdspan<Kokkos::mdspan<
            double,
            Kokkos::dextents<int, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);

    init_from_state(exec_space, prim_arrays, prim);
    init_from_state(exec_space, cons_arrays, to_cons(prim, eos.internal_energy(prim.d, prim.p)));
    exec_space.fence();

    godunov_vec(exec_space, as_const(prim_arrays), cons_arrays, eos, mesh, hllc_vec(), dt);
    exec_space.fence();

    return to_host(cons_arrays);
}

void assert_cons_near(GodunovResults const& ref, GodunovResults const& vec, double tol)
{
    ASSERT_EQ(ref.d.size(), vec.d.size());
    for (std::size_t idx = 0; idx < ref.d.size(); ++idx) {
        EXPECT_NEAR(ref.d[idx], vec.d[idx], tol) << "d   mismatch at flat index " << idx;
        EXPECT_NEAR(ref.e[idx], vec.e[idx], tol) << "e   mismatch at flat index " << idx;
        EXPECT_NEAR(ref.mx0[idx], vec.mx0[idx], tol) << "mx0 mismatch at flat index " << idx;
        EXPECT_NEAR(ref.mx1[idx], vec.mx1[idx], tol) << "mx1 mismatch at flat index " << idx;
        EXPECT_NEAR(ref.mx2[idx], vec.mx2[idx], tol) << "mx2 mismatch at flat index " << idx;
    }
}

} // namespace

TEST(GodunovRemainderWorstRem, ScalarVsVectorized)
{
    using real_t = double;
    int const n = 23;
    double const dt = 1e-9;

    Kokkos::DefaultExecutionSpace exec_space;
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);
    EulerPrim<real_t> const prim {.d = 1.0, .p = 1.0, .ux0 = 0.5, .ux1 = -0.3, .ux2 = 0.1};

    auto const ref = run_scalar(exec_space, n, prim, eos, mesh, dt);
    auto const vec = run_vec(exec_space, n, prim, eos, mesh, dt);

    assert_cons_near(ref, vec, 1e-12);
}

TEST(Godunov, ScalarVsVectorized)
{
    using real_t = double;
    int const n = 32;
    double const dt = 1e-9;

    Kokkos::DefaultExecutionSpace exec_space;
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);
    EulerPrim<real_t> const prim {.d = 1.0, .p = 1.0, .ux0 = 0.5, .ux1 = -0.3, .ux2 = 0.1};

    auto const ref = run_scalar(exec_space, n, prim, eos, mesh, dt);
    auto const vec = run_vec(exec_space, n, prim, eos, mesh, dt);

    assert_cons_near(ref, vec, 1e-12);
}

// Non-trivial flow: shock-like state with large density contrast across the domain.
// Exercises the Riemann solver more aggressively than the uniform-state tests.
TEST(GodunovShockLike, ScalarVsVectorized)
{
    using real_t = double;
    int const n = 33; // odd — also hits remainder
    double const dt = 1e-10; // smaller dt for stability with high-pressure ratio

    Kokkos::DefaultExecutionSpace exec_space;
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);
    // High-pressure, high-density state; non-zero velocities in all directions
    EulerPrim<real_t> const prim {.d = 4.0, .p = 10.0, .ux0 = 1.5, .ux1 = -0.8, .ux2 = 0.3};

    auto const ref = run_scalar(exec_space, n, prim, eos, mesh, dt);
    auto const vec = run_vec(exec_space, n, prim, eos, mesh, dt);

    assert_cons_near(ref, vec, 1e-12);
}
