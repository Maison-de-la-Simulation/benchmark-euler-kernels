#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>
#include <Kokkos_ReductionIdentity.hpp>
#include <Kokkos_SIMD.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>
#include <uniform_mesh.hpp>

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
T time_step(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        PerfectGas<T> const& eos,
        UniformMesh3d<T> const& mesh)
{
    T const invdx0 = 1 / mesh.dx0();
    T const invdx1 = 1 / mesh.dx1();
    T const invdx2 = 1 / mesh.dx2();
    T invdt {};
    EulerPrimArrays const prim_ptrs = data_handle(prim_arrays);
    Kokkos::parallel_reduce(
            "time_step_exp1_structured",
            Kokkos::RangePolicy<Kokkos::IndexType<IndexType>>(exec_space, 0, prim_arrays.d.size()),
            KOKKOS_LAMBDA(IndexType const base, T& invdt_loc) {
                EulerPrim const prim = load(prim_ptrs, base);
                T const cs = eos.speed_of_sound(prim.d, prim.p);
                T const cx0 = cs + Kokkos::abs(prim.ux0);
                T const cx1 = cs + Kokkos::abs(prim.ux1);
                T const cx2 = cs + Kokkos::abs(prim.ux2);
                T const invdt = (cx0 * invdx0) + (cx1 * invdx1) + (cx2 * invdx2);
                invdt_loc = Kokkos::max(invdt_loc, invdt);
            },
            Kokkos::Max<T>(invdt));
    return 1 / invdt;
}

namespace detail {

template <class SimdType>
struct SimdMaxReducer
{
    using reducer = SimdMaxReducer;
    using value_type = SimdType;
    using result_view_type = Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

private:
    result_view_type m_value;

public:
    KOKKOS_INLINE_FUNCTION explicit SimdMaxReducer(value_type& val) : m_value(&val) {}

    KOKKOS_INLINE_FUNCTION void join(value_type& dst, value_type const& src) const
    {
        dst = Kokkos::max(dst, src);
    }

    KOKKOS_INLINE_FUNCTION void init(value_type& val) const
    {
        using scalar_t = SimdType::value_type;
        val = value_type(Kokkos::reduction_identity<scalar_t>::max());
    }

    KOKKOS_INLINE_FUNCTION value_type& reference() const
    {
        return m_value();
    }

    KOKKOS_INLINE_FUNCTION result_view_type view() const
    {
        return m_value;
    }

    KOKKOS_INLINE_FUNCTION bool references_scalar() const
    {
        return false;
    }
};

} // namespace detail

template <class SimdType, class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
T time_step_kernel(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        PerfectGas<T> const& eos,
        UniformMesh3d<T> const& mesh,
        IndexType n0_begin,
        IndexType n0_end)
{
    constexpr IndexType width = SimdType::size();
    IndexType const n0_blocks = (n0_end - n0_begin) / width;
    IndexType const n1 = prim_arrays.d.extent(1);
    IndexType const n2 = prim_arrays.d.extent(2);

    T const invdx0 = 1 / mesh.dx0();
    T const invdx1 = 1 / mesh.dx1();
    T const invdx2 = 1 / mesh.dx2();

    Kokkos::layout_left::mapping const prim_mapping = prim_arrays.d.mapping();
    EulerPrimArrays const prim_ptrs = data_handle(prim_arrays);

    SimdType invdt_simd {};
    Kokkos::parallel_reduce(
            "time_step_vec",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(exec_space, {0, 0, 0}, {n0_blocks, n1, n2}),
            KOKKOS_LAMBDA(IndexType bi, IndexType j, IndexType k, SimdType & invdt_loc) {
                IndexType const base = prim_mapping(n0_begin + (bi * width), j, k);
                EulerPrim const prim = load<SimdType>(prim_ptrs, base);
                SimdType const cs = eos.speed_of_sound(prim.d, prim.p);
                SimdType const cx0 = cs + Kokkos::abs(prim.ux0);
                SimdType const cx1 = cs + Kokkos::abs(prim.ux1);
                SimdType const cx2 = cs + Kokkos::abs(prim.ux2);
                SimdType const invdt = (cx0 * invdx0) + (cx1 * invdx1) + (cx2 * invdx2);
                invdt_loc = Kokkos::max(invdt_loc, invdt);
            },
            detail::SimdMaxReducer<SimdType>(invdt_simd));

    return Kokkos::Experimental::reduce_max(invdt_simd);
}

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
T time_step_vec(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        PerfectGas<T> const& eos,
        UniformMesh3d<T> const& mesh)
{
    namespace KE = Kokkos::Experimental;
    using simd_t = KE::simd<T>;
    using simd_scalar_t = KE::basic_simd<T, KE::simd_abi::scalar>;

    IndexType const n0 = prim_arrays.d.extent(0);
    IndexType const vec_end = (n0 / simd_t::size()) * simd_t::size();

    T invdt = time_step_kernel<simd_t>(exec_space, prim_arrays, eos, mesh, IndexType(0), vec_end);

    if (vec_end < n0) {
        T const invdt_tail
                = time_step_kernel<simd_scalar_t>(exec_space, prim_arrays, eos, mesh, vec_end, n0);
        invdt = Kokkos::max(invdt, invdt_tail);
    }
    return 1 / invdt;
}
