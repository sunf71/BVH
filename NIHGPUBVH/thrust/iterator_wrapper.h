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

#pragma once

#include "../basic/types.h"
#include "../basic/cuda_domains.h"
#include <thrust/iterator/iterator_traits.h>

namespace nih {

/// helper wrapper for host-side thrust iterators
template <typename Space, typename Iterator>
struct iterator_wrapper
{
    typedef Iterator                                iterator_type;
    typedef typename Iterator::difference_type      difference_type;
    typedef typename Iterator::reference            reference;
    typedef typename Iterator::iterator_category    iterator_category;
    typedef iterator_wrapper<Space,Iterator>        type;

    NIH_HOST_DEVICE iterator_wrapper(const Iterator it) : m_it( it ) {}

    NIH_HOST        reference operator* () const { return *m_it; }
    NIH_HOST        reference operator[] (const difference_type n) const { return m_it[n]; }

    NIH_HOST_DEVICE type& operator+= (const difference_type n)
    {
        m_it += n;
        return *this;
    }
    NIH_HOST_DEVICE type& operator++ ()
    {
        ++m_it;
        return *this;
    }
    NIH_HOST_DEVICE type operator++ (int dummy)
    {
        type it( m_it );
        ++m_it;
        return it;
    }
    NIH_HOST_DEVICE type& operator-- ()
    {
        --m_it;
        return *this;
    }
    NIH_HOST_DEVICE type operator-- (int dummy)
    {
        type it( m_it );
        --m_it;
        return it;
    }

    NIH_HOST_DEVICE iterator_wrapper<Space,Iterator> operator+ (const difference_type n) const { return iterator_wrapper<Space,Iterator>( m_it + n ); }
    NIH_HOST_DEVICE difference_type operator- (const type it) const { return m_it - it.m_it; }

private:
    Iterator m_it;
};

#ifdef __CUDACC__
/// helper wrapper for device-side thrust iterators
template <typename Iterator>
struct iterator_wrapper<Iterator>
{
    typedef Iterator                                                            iterator_type;
    typedef typename thrust::iterator_traits<Iterator>::difference_type         difference_type;
    typedef typename thrust::iterator_traits<Iterator>::value_type              value_type;
    typedef typename thrust::detail::backend::dereference_result<Iterator>::type reference;
    typedef typename thrust::iterator_traits<Iterator>::iterator_category       iterator_category;

    typedef iterator_wrapper<Iterator> type;

    NIH_HOST_DEVICE iterator_wrapper(Iterator it) : m_it( it ) {}

    NIH_DEVICE reference operator* () const { return thrust::detail::backend::dereference( m_it ); }
    NIH_DEVICE reference operator[] (const difference_type n) const { return thrust::detail::backend::dereference( m_it, n ); }

    NIH_HOST_DEVICE type& operator+= (const difference_type n)
    {
        m_it += n;
        return *this;
    }
    NIH_HOST_DEVICE type& operator++ ()
    {
        ++m_it;
        return *this;
    }
    NIH_HOST_DEVICE type operator++ (int dummy)
    {
        type it( m_it );
        ++m_it;
        return it;
    }
    NIH_HOST_DEVICE type& operator-- ()
    {
        --m_it;
        return *this;
    }
    NIH_HOST_DEVICE type operator-- (int dummy)
    {
        type it( m_it );
        --m_it;
        return it;
    }

    NIH_HOST_DEVICE type operator+ (const difference_type n) const
    {
        return type( m_it + n );
    }
    NIH_HOST_DEVICE difference_type operator- (const type it) const { return m_it - it.m_it; }

private:
    Iterator m_it;
};
#endif


template <typename Iterator>
iterator_wrapper<
    typename thrust::iterator_space<Iterator>::type,
    Iterator>
wrap(Iterator it)
{
    return iterator_wrapper<
        typename thrust::iterator_space<Iterator>::type,
        Iterator>( it );
}

template <typename Space, typename Iterator>
Iterator unwrap(const iterator_wrapper<Space,Iterator> it) { return it.m_it; }

} // namespace nih
