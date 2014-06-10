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

/*! \file scan.h
 *   \brief Define CUDA based scan primitives.
 */

#pragma once

#include "basic/types.h"

namespace nih {
namespace cuda {

/// intra-warp inclusive scan
///
/// \param val      per-threrad input value
/// \param tidx     warp thread index
/// \param red      scan result storage (2*WARP_SIZE elements)
template <typename T> inline __device__ __forceinline__ T scan_warp(T val, const int32 tidx, volatile T *red)
{
    // pad initial segment with zeros
    red[tidx] = 0;
    red += 32;

    // Hillis-Steele scan
    red[tidx] = val;
    val += red[tidx-1];  red[tidx] = val;
    val += red[tidx-2];  red[tidx] = val;
    val += red[tidx-4];  red[tidx] = val;
    val += red[tidx-8];  red[tidx] = val;
    val += red[tidx-16]; red[tidx] = val;
	return val;
}
/// return the total from a scan_warp
///
/// \param red      scan result storage
template <typename T> inline __device__ __forceinline__ T scan_warp_total(volatile T *red) { return red[63]; }

/// alloc n elements per thread from a common pool, using a synchronous warp scan
///
/// \param n                number of elements to alloc
/// \param warp_tid         warp thread index
/// \param warp_red         temporary warp scan storage (2*WARP_SIZE elements)
/// \param warp_broadcast   temporary warp broadcasting storage
__device__ __forceinline__
uint32 alloc(uint32 n, uint32* pool, const int32 warp_tid, volatile uint32* warp_red, volatile uint32* warp_broadcast)
{
    uint32 warp_scan  = scan_warp( n, warp_tid, warp_red ) - n;
    uint32 warp_count = scan_warp_total( warp_red );
    if (warp_tid == 0)
        *warp_broadcast = atomicAdd( pool, warp_count );

    return *warp_broadcast + warp_scan;
}

/// alloc zero or exactly N elements per thread from a common pool
///
/// \param p                allocation predicate
/// \param warp_tid         warp thread id
/// \param warp_broadcast   temporary warp broadcasting storage
template <uint32 N>
__device__ __forceinline__
uint32 alloc(bool p, uint32* pool, const int32 warp_tid, volatile uint32* warp_broadcast)
{
    const uint32 warp_mask  = __ballot( p );
    const uint32 warp_count = __popc( warp_mask );
    const uint32 warp_scan  = __popc( warp_mask << (warpSize - warp_tid) );

    // acquire an offset for this warp
    if (warp_scan == 0 && p)
        *warp_broadcast = atomicAdd( pool, warp_count * N );

    // find offset
    return *warp_broadcast + warp_scan * N;
}

} // namespace cuda
} // namespace nih
