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
#include "../../basic/types.h"
#include "../../basic/numbers.h"
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>


namespace nih {
namespace cuda {

namespace sah {
	template <typename keyType, typename ValType>
	void radix_sort(
		const uint32    n_elements,
		keyType*         keys,
		ValType*         values,
		keyType*         keys_tmp,
		ValType*         values_tmp)
	{
		using namespace cub;
		cub::DoubleBuffer<keyType> d_keys;
		d_keys.d_buffers[0] = keys;
		d_keys.d_buffers[1] = keys_tmp;
		cub::DoubleBuffer<ValType> d_values;
		d_values.d_buffers[0] = values;
		d_values.d_buffers[1] = values_tmp;

		// Allocate temporary storage
		size_t  temp_storage_bytes  = 0;
		void    *d_temp_storage     = NULL;
		CachingDeviceAllocator  allocator(true); 
		CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n_elements));
		CubDebugExit(allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
		CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, n_elements));

		{
			cudaMemcpy( keys,   keys_tmp,   sizeof(keyType)*n_elements, cudaMemcpyDeviceToDevice );
			cudaMemcpy( values, values_tmp, sizeof(ValType)*n_elements, cudaMemcpyDeviceToDevice );
		}
	}
//void radix_sort(
//    const uint32    n_elements,
//    uint32*         keys,
//    uint32*         values,
//    uint32*         keys_tmp,
//    uint32*         values_tmp)
//{
//    b40c::util::PingPongStorage<uint32, uint32> sort_storage( keys, keys_tmp, values, values_tmp );
//
//    b40c::radix_sort::Enactor enactor;
//
//    if (n_elements < 1024*1024)
//        enactor.Sort<b40c::radix_sort::SMALL_SIZE>( sort_storage, n_elements );
//    else
//        enactor.Sort<b40c::radix_sort::LARGE_SIZE>( sort_storage, n_elements );
//
//    if (sort_storage.selector)
//    {
//        cudaMemcpy( keys,   keys_tmp,   sizeof(uint32)*n_elements, cudaMemcpyDeviceToDevice );
//        cudaMemcpy( values, values_tmp, sizeof(uint32)*n_elements, cudaMemcpyDeviceToDevice );
//    }
//}
//void radix_sort(
//    const uint32    n_elements,
//    uint64*         keys,
//    uint32*         values,
//    uint64*         keys_tmp,
//    uint32*         values_tmp)
//{
//    b40c::util::PingPongStorage<uint64, uint32> sort_storage( keys, keys_tmp, values, values_tmp );
//
//    b40c::radix_sort::Enactor enactor;
//
//    if (n_elements < 1024*1024)
//        enactor.Sort<b40c::radix_sort::SMALL_SIZE>( sort_storage, n_elements );
//    else
//        enactor.Sort<b40c::radix_sort::LARGE_SIZE>( sort_storage, n_elements );
//
//    if (sort_storage.selector)
//    {
//        cudaMemcpy( keys,   keys_tmp,   sizeof(uint64)*n_elements, cudaMemcpyDeviceToDevice );
//        cudaMemcpy( values, values_tmp, sizeof(uint32)*n_elements, cudaMemcpyDeviceToDevice );
//    }
//}

} // namespace sah

} // namespace cuda
} // namespace nih
