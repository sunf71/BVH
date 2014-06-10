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

#include <stddef.h>

namespace nih {

template <typename T, typename Domain>
struct domain_pointer
{
    typedef int difference_type;

    domain_pointer() : m_ptr(NULL) {}
    explicit domain_pointer(T* p) : m_ptr(p) {}

    template <typename U>
    domain_pointer(U* p) : m_ptr(p) {}

    template <typename U>
    domain_pointer(const domain_pointer<U,Domain> p) : m_ptr(p.ptr()) {}

    T* ptr() const { return m_ptr; }

private:
    T* m_ptr;
};

template <typename T, typename Domain>
typename domain_pointer<T,Domain>::difference_type
operator- (const domain_pointer<T,Domain> p1, const domain_pointer<T,Domain> p2) { return p1.ptr() - p2.ptr(); }

template <typename T, typename Domain>
typename domain_pointer<T,Domain>::difference_type
operator- (const domain_pointer<T,Domain> p1, const domain_pointer<const T,Domain> p2) { return p1.ptr() - p2.ptr(); }

template <typename T, typename Domain>
typename domain_pointer<T,Domain>::difference_type
operator- (const domain_pointer<const T,Domain> p1, const domain_pointer<T,Domain> p2) { return p1.ptr() - p2.ptr(); }

template <typename T, typename Domain>
domain_pointer<T,Domain> operator+ (const domain_pointer<T,Domain> p1, int d) { return domain_pointer<T,Domain>( p1.ptr() + d ); }

template <typename T, typename Domain>
domain_pointer<T,Domain> operator+ (const domain_pointer<T,Domain> p1, size_t d) { return domain_pointer<T,Domain>( p1.ptr() + d ); }

template <typename T, typename U, typename Domain>
domain_pointer<T,Domain> domain_pointer_cast(const domain_pointer<U,Domain> p) { return domain_pointer<T,Domain>( reinterpret_cast<T*>( p.ptr() ) ); }

} // namespace nih
