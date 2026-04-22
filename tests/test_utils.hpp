#pragma once

#include <gtest/gtest.h>

#include <euler_arrays.hpp>


template <class T>
void compare(EulerPrimArrays<T> const& ref, EulerPrimArrays<T> const& vec, double tol, int idx)
{
    ASSERT_NEAR(ref.d(idx), vec.d(idx), tol);
    ASSERT_NEAR(ref.p(idx), vec.p(idx), tol);
    ASSERT_NEAR(ref.ux0(idx), vec.ux0(idx), tol);
    ASSERT_NEAR(ref.ux1(idx), vec.ux1(idx), tol);
    ASSERT_NEAR(ref.ux2(idx), vec.ux2(idx), tol);
}

template <class T>
void compare(EulerConsArrays<T> const& ref, EulerConsArrays<T> const& vec, double tol, int idx)
{
    ASSERT_NEAR(ref.d(idx), vec.d(idx), tol);
    ASSERT_NEAR(ref.e(idx), vec.e(idx), tol);
    ASSERT_NEAR(ref.mx0(idx), vec.mx0(idx), tol);
    ASSERT_NEAR(ref.mx1(idx), vec.mx1(idx), tol);
    ASSERT_NEAR(ref.mx2(idx), vec.mx2(idx), tol);
}
