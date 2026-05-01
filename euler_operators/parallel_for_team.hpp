#pragma once

#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

namespace detail {

template <typename IndexType, typename Functor>
class EulerParallelForWrapper
{
    Kokkos::Array<IndexType, 3> m_begin;

    Kokkos::Array<IndexType, 3> m_end;

    Functor m_functor;

    KOKKOS_FUNCTION Kokkos::Array<IndexType, 2> team_to_coord(IndexType id) const noexcept
    {
        IndexType const size1 = m_end[1] - m_begin[1];
        IndexType const i2 = id / size1;
        IndexType const i1 = id - (size1 * i2);
        return Kokkos::Array<IndexType, 2> {i1 + m_begin[1], i2 + m_begin[2]};
    }

public:
    EulerParallelForWrapper(
            Kokkos::Array<IndexType, 3> const& begin,
            Kokkos::Array<IndexType, 3> const& end,
            Functor functor)
        : m_begin(begin)
        , m_end(end)
        , m_functor(std::move(functor))
    {
    }

    KOKKOS_FUNCTION void operator()(auto const& team_member) const noexcept
    {
        auto const [i1, i2] = team_to_coord(team_member.league_rank());
        Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, m_begin[0], m_end[0]),
                [&](IndexType i0) { m_functor(i0, i1, i2); });
    }
};

} // namespace detail

template <typename ExecutionSpace, typename IndexType, typename Functor>
void euler_parallel_for(
        std::string const& str,
        ExecutionSpace const& execution_space,
        Kokkos::Array<IndexType, 3> const& begin,
        Kokkos::Array<IndexType, 3> const& end,
        Functor const& functor)
{
    Kokkos::TeamPolicy const policy(execution_space, (end[1] - begin[1]) * (end[2] - begin[2]), 32);
    Kokkos::parallel_for(str, policy, detail::EulerParallelForWrapper(begin, end, functor));
}
