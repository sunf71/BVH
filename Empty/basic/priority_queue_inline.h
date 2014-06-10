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

#include <nih/basic/types.h>

namespace nih {

// constructor
//
template <typename Key, typename Compare, uint32 SIZE>
FORCE_INLINE NIH_HOST_DEVICE priority_queue<Key,Compare,SIZE>::priority_queue(const Compare cmp)
    : m_size(0), m_cmp(cmp)
{
}

// is queue empty?
//
template <typename Key, typename Compare, uint32 SIZE>
FORCE_INLINE NIH_HOST_DEVICE bool priority_queue<Key,Compare,SIZE>::empty() const
{
    return m_size == 0;
}

// return queue size
//
template <typename Key, typename Compare, uint32 SIZE>
FORCE_INLINE NIH_HOST_DEVICE uint32 priority_queue<Key,Compare,SIZE>::size() const
{
    return m_size;
}

// push an element
//
template <typename Key, typename Compare, uint32 SIZE>
FORCE_INLINE NIH_HOST_DEVICE void priority_queue<Key,Compare,SIZE>::push(const Key key)
{
    // check whether the queue is full
	if (m_size >= SIZE)
		pop();

    m_size++;

	uint32 r = m_size;
	while (r > 1) // sift up new item
	{		
		const uint32 p = r/2;
        if (! m_cmp( m_queue[p], key )) // in proper order
			break;

        m_queue[r] = m_queue[p]; // else swap with parent
		r = p;
	}
    m_queue[r] = key; // insert new item at final location
}

// pop an element
//
template <typename Key, typename Compare, uint32 SIZE>
FORCE_INLINE NIH_HOST_DEVICE void priority_queue<Key,Compare,SIZE>::pop()
{
	Key dn = m_queue[m_size--];  // last item in queue
	uint32 p = 1;                // p points to item out of position
	uint32 r = p << 1;           // left child of p

	while (r <= m_size) // while r is still within the heap
	{
		// set r to smaller child of p
		if (r < m_size  && m_cmp( m_queue[r], m_queue[r+1] )) r++;
		if (! m_cmp( dn, m_queue[r] ))	// in proper order
			break;

		m_queue[p] = m_queue[r];    // else swap with child
		p = r;                      // advance pointers
		r = p<<1;
	}
    m_queue[p] = m_queue[m_size+1]; // insert last item in proper place
}

// top of the queue
//
template <typename Key, typename Compare, uint32 SIZE>
FORCE_INLINE NIH_HOST_DEVICE Key priority_queue<Key,Compare,SIZE>::top() const
{
    return m_queue[1];
}

// return the i-th element in the heap
//
template <typename Key, typename Compare, uint32 SIZE>
FORCE_INLINE NIH_HOST_DEVICE const Key& priority_queue<Key,Compare,SIZE>::operator[] (const uint32 i) const
{
    return m_queue[1+i];
}

} // namespace nih
