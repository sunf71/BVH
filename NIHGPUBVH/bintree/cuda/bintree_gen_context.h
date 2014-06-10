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

/*! \file bintree_gen_context.h
 *   \brief Defines the context class for the binary tree generate() function.
 */

#pragma once

#include "../../basic/types.h"
#include <thrust/device_vector.h>

namespace nih {
namespace cuda {

/*! \addtogroup bintree Binary Trees
 *  \{
 */

///
/// A context class for binary tree generate() function.
///
struct Bintree_gen_context
{
    struct Split_task
    {
        NIH_HOST_DEVICE Split_task() {}
        NIH_HOST_DEVICE Split_task(const uint32 id, const uint32 begin, const uint32 end, const uint32 in = 0)
            : m_node( id ), m_begin( begin ), m_end( end ), m_input( in ) {}

        uint32 m_node;
        uint32 m_begin;
        uint32 m_end;
        uint32 m_input;
    };

    thrust::device_vector<Split_task>   m_task_queues[2];
    thrust::device_vector<uint32>       m_counters;
    thrust::device_vector<uint32>       m_skip_nodes;
    uint32                              m_nodes;
    uint32                              m_leaves;
    uint32                              m_levels[64];
};

/*! \}
 */

} // namespace cuda
} // namespace nih
