/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*! \file functors.h
 *   \brief Defines some general purpose functors.
 */

#pragma once

#include "../../basic/types.h"
#include "../../linalg/vector.h"

namespace nih {

struct unary_function_tag {};
struct binary_function_tag {};
struct ternary_function_tag {};

///
/// default predicate functor, returning the standard conversion to boolean
///
template <typename T>
struct default_predicate
{
    typedef unary_function_tag function_tag;

    NIH_HOST_DEVICE bool operator() (const T t) const { return t ? true : false; }
};

///
/// constant functor
///
template <typename T, typename R>
struct constant_fun
{
    typedef unary_function_tag function_tag;
    typedef T argument_type;
    typedef R result_type;

    /// constructor
    ///
    /// \param c    constant value
    constant_fun(R c) : constant(c) {}

    NIH_HOST_DEVICE R operator() (const T op) const { return constant; }

    R constant;
};
///
/// A functor to return the constant 1 cast to a given type
///
template <typename T, typename R>
struct one_fun
{
    typedef unary_function_tag function_tag;
    typedef T argument_type;
    typedef R result_type;

    NIH_HOST_DEVICE R operator() (const T op) const { return R(1); }
};
///
/// A functor to return the either one or zero depending on the
/// boolean predicate evaluation of the input.
///
struct one_or_zero
{
    typedef unary_function_tag function_tag;
    typedef uint32 argument_type;
    typedef uint32 result_type;

    NIH_HOST_DEVICE uint32 operator() (const uint32 op) const
    {
        return op ? 1u : 0u;
    }
};
///
/// A functor to negate the input value
///
template <typename T>
struct not
{
    typedef unary_function_tag function_tag;
    typedef T argument_type;
    typedef T result_type;

    NIH_HOST_DEVICE T operator() (const T op) const { return !op; }
};
///
/// A functor to return the input value minus 1
///
template <typename T>
struct minus_one
{
    typedef unary_function_tag function_tag;
    typedef T      argument_type;
    typedef T      result_type;

    NIH_HOST_DEVICE T operator() (const T op) const { return op - T(1); }
};
///
/// A functor to bind the first argument of a binary operator to a constant
///
template <typename F, typename C>
struct binder1st
{
    typedef unary_function_tag function_tag;
    typedef typename F::first_argument_type argument_type;
    typedef typename F::result_type         result_type;

    binder1st(const F& f, const C c) : functor(f), first(c) {}

    NIH_HOST_DEVICE uint32 operator() (const uint32 op) const { return functor( first, op ); }

    F functor;
    C first;
};
///
/// A functor to bind the second argument of a binary operator to a constant
///
template <typename F, typename C>
struct binder2nd
{
    typedef unary_function_tag function_tag;
    typedef typename F::second_argument_type argument_type;
    typedef typename F::result_type          result_type;

    binder2nd(const F& f, const C c) : functor(f), second(c) {}

    NIH_HOST_DEVICE uint32 operator() (const uint32 op) const { return functor( op, second ); }

    F functor;
    C second;
};
template <typename F, typename C> binder1st<F,C> bind1st(const F& f, const C c) { return binder1st<F,C>( f, c ); }
template <typename F, typename C> binder2nd<F,C> bind2nd(const F& f, const C c) { return binder2nd<F,C>( f, c ); }

///
/// return the second_argument component of the first_argument vector
///
template <typename Vector_type>
struct component_functor
{
    typedef binary_function_tag function_tag;
    typedef Vector_type                         first_argument_type;
    typedef uint32                              second_argument_type;
    typedef typename Vector_type::value_type    result_type;

    NIH_HOST_DEVICE result_type operator() (const first_argument_type v, const second_argument_type i) const { return v[i]; }
};
///
/// square functor
///
template <typename T>
struct sqr_functor
{
    typedef unary_function_tag function_tag;
    typedef T   argument_type;
    typedef T   result_type;

    NIH_HOST_DEVICE T operator() (const T& v) const { return v*v; }
};
///
/// greater than zero functor
///
template <typename T>
struct greater_than_zero
{
    typedef unary_function_tag function_tag;
    typedef T       argument_type;
    typedef bool    result_type;

    NIH_HOST_DEVICE bool operator() (const T& v) const { return v > 0; }
};
///
/// equal to functor
///
template <typename T>
struct equal_to
{
    typedef binary_function_tag function_tag;
    typedef T       first_argument_type;
    typedef T       second_argument_type;
    typedef bool    result_type;

    NIH_HOST_DEVICE bool operator() (const T& op1, const T& op2) const { return op1 == op2; }
};
///
/// not equal to functor
///
template <typename T>
struct not_equal_to
{
    typedef binary_function_tag function_tag;
    typedef T       first_argument_type;
    typedef T       second_argument_type;
    typedef bool    result_type;

    NIH_HOST_DEVICE bool operator() (const T& op1, const T& op2) const { return op1 != op2; }
};
///
/// A functor to compute equality to a given constant
///
template <typename T>
struct eq_constant
{
    typedef unary_function_tag function_tag;
    typedef T       argument_type;
    typedef bool    result_type;

    /// constructor
    ///
    /// \param c    constant value
    eq_constant(const T c) : m_c(c) {}

    NIH_HOST_DEVICE bool operator() (const T& v) const { return v == m_c; }

private:
    const T m_c;
};
///
/// A functor to compute inequality to a given constant
///
template <typename T>
struct neq_constant
{
    typedef unary_function_tag function_tag;
    typedef T       argument_type;
    typedef bool    result_type;

    /// constructor
    ///
    /// \param c    constant value
    neq_constant(const T c) : m_c(c) {}

    NIH_HOST_DEVICE bool operator() (const T& v) const { return v != m_c; }

private:
    const T m_c;
};
///
/// if true functor
///
template <typename T, typename R>
struct if_true
{
    typedef unary_function_tag function_tag;
    typedef T       argument_type;
    typedef R       result_type;

    /// constructor
    ///
    /// \param r0   true output value
    /// \param r1   false output value
    if_true(const R r0, const R r1) : m_r_true(r0), m_r_false(r1) {}

    NIH_HOST_DEVICE R operator() (const T& v) const { return v ? m_r_true : m_r_false; }

private:
    const R m_r_true;
    const R m_r_false;
};
///
/// A functor to select between two output values based on equality to a given constant
///
template <typename T, typename R>
struct if_constant
{
    typedef unary_function_tag function_tag;
    typedef T       argument_type;
    typedef R       result_type;

    /// constructor
    ///
    /// \param c    constant value
    /// \param r0   true output value
    /// \param r1   false output value
    if_constant(const T c, const R r0, const R r1) : m_c(c), m_r_true(r0), m_r_false(r1) {}

    NIH_HOST_DEVICE R operator() (const T& v) const { return v == m_c ? m_r_true : m_r_false; }

private:
    const T m_c;
    const R m_r_true;
    const R m_r_false;
};
///
/// compose two unary functions
///
template <typename F1, typename F2>
struct compose_unary
{
    typedef unary_function_tag                 function_tag;
    typedef typename F2::first_argument_type   argument_type;
    typedef typename F1::result_type           result_type;

    compose_unary(const F1 f1, const F2 f2) : m_fun1(f1), m_fun2(f2) {}

    NIH_HOST_DEVICE result_type operator() (const argument_type& op) const { return m_fun1( m_fun2( op ) ); }

private:
    const F1 m_fun1;
    const F2 m_fun2;
};
///
/// compose a binary function after two unary ones
///
template <typename F, typename G1, typename G2>
struct compose_binary
{
    typedef binary_function_tag                function_tag;
    typedef typename G1::argument_type         first_argument_type;
    typedef typename G2::argument_type         second_argument_type;
    typedef typename F::result_type            result_type;

    compose_binary(const F f, const G1 g1, const G2 g2) : m_f(f), m_g1(g1), m_g2(g2) {}

    NIH_HOST_DEVICE result_type operator() (
        const first_argument_type&  op1,
        const second_argument_type& op2) const { return m_f( m_g1( op1 ), m_g2( op2 ) ); }

private:
    const F  m_f;
    const G1 m_g1;
    const G2 m_g2;
};
///
/// compose an unary operator F1 with a binary operator F2
///
template <typename F1, typename F2>
struct compose_unary_after_binary
{
    typedef binary_function_tag                function_tag;
    typedef typename F2::first_argument_type   first_argument_type;
    typedef typename F2::second_argument_type  second_argument_type;
    typedef typename F1::result_type           result_type;

    compose_unary_after_binary(const F1 f1, const F2 f2) : m_fun1(f1), m_fun2(f2) {}

    NIH_HOST_DEVICE result_type operator() (
        const first_argument_type&  op1,
        const second_argument_type& op2) const { return m_fun1( m_fun2( op1, op2 ) ); }

private:
    const F1 m_fun1;
    const F2 m_fun2;
};

template <typename F1, typename F2, typename T1, typename T2>
struct composition_type {};
template <typename F1, typename F2>
struct composition_type<F1,F2,unary_function_tag,binary_function_tag> { typedef compose_unary_after_binary<F1,F2> type; };
template <typename F1, typename F2>
struct composition_type<F1,F2,unary_function_tag,unary_function_tag> { typedef compose_unary<F1,F2> type; };

/// compose two functions
///
template <typename F1, typename F2>
typename composition_type<F1,F2,typename F1::function_tag,typename F2::function_tag>::type compose(const F1 f1, const F2 f2)
{
    return composition_type<F1,F2,typename F1::function_tag,typename F2::function_tag>::type( f1, f2 );
}
/// compose a binary function after two unary ones
///
template <typename F, typename G1, typename G2>
compose_binary<F,G1,G2> compose(const F f, const G1 g1, const G2 g2)
{
    return compose_binary<F,G1,G2>( f, g1, g2 );
}

///
/// minimum functor
///
template <typename T>
struct min_functor
{
    typedef T   first_argument_type;
    typedef T   second_argument_type;
    typedef T   result_type;

    NIH_HOST_DEVICE T operator() (const T a, const T b) const { return a < b ? a : b; }
};
///
/// maximum functor
///
template <typename T>
struct max_functor
{
    typedef T   first_argument_type;
    typedef T   second_argument_type;
    typedef T   result_type;

    NIH_HOST_DEVICE T operator() (const T a, const T b) const { return a > b ? a : b; }
};

///
/// addition functor
///
template <typename T>
struct add
{
    typedef T      first_argument_type;
    typedef T      second_argument_type;
    typedef T      result_type;

    NIH_HOST_DEVICE T operator() (const T op1, const T op2) const { return op1 + op2; }
};
///
/// binary OR functor
///
template <typename T>
struct binary_or
{
    typedef T      first_argument_type;
    typedef T      second_argument_type;
    typedef T      result_type;

    NIH_HOST_DEVICE T operator() (const T op1, const T op2) const { return op1 | op2; }
};
///
/// binary AND functor
///
template <typename T>
struct binary_and
{
    typedef T      first_argument_type;
    typedef T      second_argument_type;
    typedef T      result_type;

    NIH_HOST_DEVICE T operator() (const T op1, const T op2) const { return op1 & op2; }
};

///
/// A functor to compute the binary AND with a mask
///
template <typename T>
struct mask_and
{
    typedef T      argument_type;
    typedef T      result_type;

    /// constructor
    ///
    /// \param mask     mask value
    NIH_HOST_DEVICE mask_and(const T mask) : m_mask( mask ) {}

    NIH_HOST_DEVICE T operator() (const T op) const { return op & m_mask; }

private:
    const T m_mask;
};

///
/// A functor to compute the binary OR with a mask
///
template <typename T>
struct mask_or
{
    typedef T      argument_type;
    typedef T      result_type;

    /// constructor
    ///
    /// \param mask     mask value
    NIH_HOST_DEVICE mask_or(const T mask) : m_mask( mask ) {}

    NIH_HOST_DEVICE T operator() (const T op) const { return op | m_mask; }

private:
    const T m_mask;
};

///
/// A functor to shift values to the left by a given amount of bits
///
template <typename T>
struct l_bit_shift
{
    typedef T argument_type;
    typedef T result_type;

    /// constructor
    ///
    /// \param bits     shift size
    NIH_HOST_DEVICE l_bit_shift(const T bits) : m_bits( bits ) {}

    NIH_HOST_DEVICE  T operator() (const T x) const { return x << m_bits; }

private:
    const T m_bits;
};

///
/// A functor to shift values to the left by a given amount of bits
///
template <typename T>
struct r_bit_shift
{
    typedef T argument_type;
    typedef T result_type;

    /// constructor
    ///
    /// \param bits     shift size
    NIH_HOST_DEVICE r_bit_shift(const T bits) : m_bits( bits ) {}

    NIH_HOST_DEVICE  T operator() (const T x) const { return x >> m_bits; }

private:
    const T m_bits;
};

///
/// A functor to compute the clamped cosine of the angle formed with
/// a given normal
///
template <typename Vector_type>
struct clamped_cosine_fun
{
    typedef Vector_type argument_type;
    typedef float       result_type;

    /// constructor
    ///
    /// \param normal       reference normal
    NIH_HOST_DEVICE clamped_cosine_fun(const Vector_type& normal) : m_normal( normal ) {}

    NIH_HOST_DEVICE float operator() (const Vector_type& dir) const { return max( dot( dir, m_normal ), 0.0f ); }

    const Vector_type m_normal;
};

///
/// A functor to compute the absolute value of the cosine of the angle formed with
/// a given normal
///
template <typename Vector_type>
struct abs_cosine_fun
{
    typedef Vector_type argument_type;
    typedef float       result_type;

    /// constructor
    ///
    /// \param normal       reference normal
    NIH_HOST_DEVICE abs_cosine_fun(const Vector_type& normal) : m_normal( normal ) {}

    NIH_HOST_DEVICE float operator() (const Vector_type& dir) const { return fabsf( dot( dir, m_normal ) ); }

    const Vector_type m_normal;
};

} // namespace nih
