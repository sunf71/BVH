//
// C++ class template imprementation of SFMT(SSE2)
//
// Copyright (c) 2008 Tripcode Explorer Project. Rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Original imprementation of SFMT (SIMD-oriented Fast Mersenne Twister)
//   Copyright (C) 2006, 2007 Mutsuo Saito, Makoto Matsumoto and
//   Hiroshima University. All rights reserved.
//   (see original/LICENSE.txt)

#ifndef SSE2_SFMT_PLUS__
#define SSE2_SFMT_PLUS__

namespace sfmtrand {
static const char copyright[] = "SFMT implementation Copyright (c) 2008 Tripcode Explorer Project. Rights reserved.\n";
}

#include <emmintrin.h>
#include <cstdio>
#include <cassert>
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
  #include <inttypes.h>
#elif defined(_MSC_VER) || defined(__BORLANDC__)
  typedef unsigned int uint32_t;
  typedef unsigned __int64 uint64_t;
  #define forceinline __inline
#else
  #include <inttypes.h>
  #if defined(__GNUC__)
    #define forceinline __inline__
  #endif
#endif

namespace sfmtplus { 
	template < uint32_t mexp, uint32_t pos1, 
		uint32_t sl1, uint32_t sl2, uint32_t sr1, uint32_t sr2, 
		uint32_t msk1, uint32_t msk2, uint32_t msk3, uint32_t msk4,
		uint32_t parity1, uint32_t parity2, uint32_t parity3, uint32_t parity4 >
	class sse2_fast_mersenne_twister {
	public:
		enum { 
			MEXP = mexp,
			POS1 = pos1, 
			SL1 = sl1, 
			SL2 = sl2,
			SR1 = sr1,
			SR2 = sr2, 
			MSK1 = msk1,
			MSK2 = msk2,
			MSK3 = msk3,
			MSK4 = msk4,
			PARITY1 = parity1, 
			PARITY2 = parity2, 
			PARITY3 = parity3, 
			PARITY4 = parity4,
			N = MEXP / 128 + 1,
			N32 = N * 4,
			N64 = N * 2
		};

	protected:
		forceinline static double to_real2(uint32_t v) { return v * (1.0/4294967296.0); }
		forceinline static double to_real1(uint32_t v) { return v * (1.0/4294967295.0); }
		forceinline static double to_real3(uint32_t v) { return (((double)v) + 0.5)*(1.0/4294967296.0); }
		forceinline static double to_res53(uint64_t v) { return v * (1.0/18446744073709551616.0L); };
		forceinline static double to_res53_mix(uint32_t x, uint32_t y) { return to_res53(x | ((uint64_t)y << 32)); }
		forceinline static int idxof(int i) { return i; }
		forceinline static uint32_t func1(uint32_t x) { return (x ^ (x >> 27)) * (uint32_t)1664525UL; }
		forceinline static uint32_t func2(uint32_t x) { return (x ^ (x >> 27)) * (uint32_t)1566083941UL; }

		union w128_t {
			__m128i si;
			uint32_t u[4];
		};

		w128_t sfmt[N];
		uint32_t *psfmt32;
		uint64_t *psfmt64;
		int idx;
		int initialized;
		uint32_t parity[4];
		char idstring[80];

		static __m128i mm_recursion(__m128i *a, const __m128i* b, __m128i c,
			__m128i d, __m128i mask) {
			__m128i v, x, y, z;
			x = _mm_load_si128(a);
			y = _mm_srli_epi32(*b, SR1);
			z = _mm_srli_si128(c, SR2);
			v = _mm_slli_epi32(d, SL1);
			z = _mm_xor_si128(z, x);
			z = _mm_xor_si128(z, v);
			x = _mm_slli_si128(x, SL2);
			y = _mm_and_si128(y, mask);
			z = _mm_xor_si128(z, x);
			z = _mm_xor_si128(z, y);
			return z;
		}
		void period_certification(void) {
			int inner = 0;
			int i, j;
			uint32_t work;

			for (i = 0; i < 4; i++)
				inner ^= psfmt32[idxof(i)] & parity[i];
			for (i = 16; i > 0; i >>= 1)
				inner ^= inner >> i;
			inner &= 1;
			/* check OK */
			if (inner == 1) {
				return;
			}
			/* check NG, and modification */
			for (i = 0; i < 4; i++) {
				work = 1;
				for (j = 0; j < 32; j++) {
					if ((work & parity[i]) != 0) {
						psfmt32[idxof(i)] ^= work;
						return;
					}
					work = work << 1;
				}
			}
		}
	public:
		sse2_fast_mersenne_twister(void) {
			psfmt32 = &sfmt[0].u[0];
			psfmt64 = (uint64_t *)&sfmt[0].u[0];
			initialized = 0;
			parity[0] = PARITY1; 
			parity[1] = PARITY2;
			parity[2] = PARITY3;
			parity[3] = PARITY4;
			std::sprintf(idstring, "SFMT-%d:%d-%d-%d-%d-%d:%08x-%08x-%08x-%08x",
				MEXP, POS1, SL1, SL2, SR1, SR2, MSK1, MSK2, MSK3, MSK4);
		}
		sse2_fast_mersenne_twister(uint32_t seed) {
			sse2_fast_mersenne_twister();
			init_gen_rand(seed);
		}
		sse2_fast_mersenne_twister(uint32_t *init_key, int key_length) {
			sse2_fast_mersenne_twister();
 			init_by_array(init_key, key_length);
		}
		void gen_rand_all(void) {
			int i;
			__m128i r, r1, r2;
			/*static*/ const __m128i mask = _mm_set_epi32( MSK4, MSK3, MSK2, MSK1 );

			r1 = _mm_load_si128(&sfmt[N - 2].si);
			r2 = _mm_load_si128(&sfmt[N - 1].si);
			for (i = 0; i < N - POS1; i++) {
				r = mm_recursion(&sfmt[i].si, &sfmt[i + POS1].si, r1, r2, mask);
				_mm_store_si128(&sfmt[i].si, r);
				r1 = r2;
				r2 = r;
			}
			for (; i < N; i++) {
				r = mm_recursion(&sfmt[i].si, &sfmt[i + POS1 - N].si, r1, r2, mask);
				_mm_store_si128(&sfmt[i].si, r);
				r1 = r2;
				r2 = r;
			}
		}
		void gen_rand_array(w128_t *array, int size) {
			int i, j;
			__m128i r, r1, r2;
			static const __m128i mask = _mm_set_epi32(MSK4, MSK3, MSK2, MSK1);
	
			r1 = _mm_load_si128(&sfmt[N - 2].si);
			r2 = _mm_load_si128(&sfmt[N - 1].si);
			for (i = 0; i < N - POS1; i++) {
				r = mm_recursion(&sfmt[i].si, &sfmt[i + POS1].si, r1, r2, mask);
				_mm_store_si128(&array[i].si, r);
				r1 = r2;
				r2 = r;
			}
			for (; i < N; i++) {
				r = mm_recursion(&sfmt[i].si, &array[i + POS1 - N].si, r1, r2, mask);
				_mm_store_si128(&array[i].si, r);
				r1 = r2;
				r2 = r;
			}
			/* main loop */
			for (; i < size - N; i++) {
				r = mm_recursion(&array[i - N].si, &array[i + POS1 - N].si, r1, r2,
					mask);
				_mm_store_si128(&array[i].si, r);
				r1 = r2;
				r2 = r;
			}
			for (j = 0; j < 2 * N - size; j++) {
				r = _mm_load_si128(&array[j + size - N].si);
				_mm_store_si128(&sfmt[j].si, r);
			}
			for (; i < size; i++) {
				r = mm_recursion(&array[i - N].si, &array[i + POS1 - N].si, r1, r2,
					mask);
				_mm_store_si128(&array[i].si, r);
				_mm_store_si128(&sfmt[j++].si, r);
				r1 = r2;
				r2 = r;
			}
		}
		void init_gen_rand(uint32_t seed) {
			int i;
			psfmt32[idxof(0)] = seed;
			for (i = 1; i < N32; i++) {
				psfmt32[idxof(i)] = 1812433253UL * (psfmt32[idxof(i - 1)]
				^ (psfmt32[idxof(i - 1)] >> 30)) + i;
			}
			idx = N32;
			period_certification();
			initialized = 1;
		}
		void init_by_array(uint32_t *init_key, int key_length) {
			int i, j, count;
			uint32_t r;
			int lag;
			int mid;
			int size = N * 4;

			if (size >= 623) {
				lag = 11;
			} else if (size >= 68) {
				lag = 7;
			} else if (size >= 39) {
				lag = 5;
			} else {
				lag = 3;
			}
			mid = (size - lag) / 2;

			memset(sfmt, 0x8b, sizeof(sfmt));
			if (key_length + 1 > N32) {
				count = key_length + 1;
			} else {
				count = N32;
			}
			r = func1(psfmt32[idxof(0)] ^ psfmt32[idxof(mid)] ^ psfmt32[idxof(N32 - 1)]);
			psfmt32[idxof(mid)] += r;
			r += key_length;
			psfmt32[idxof(mid + lag)] += r;
			psfmt32[idxof(0)] = r;

			count--;
			for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
				r = func1(psfmt32[idxof(i)] ^ psfmt32[idxof((i + mid) % N32)]
					^ psfmt32[idxof((i + N32 - 1) % N32)]);
				psfmt32[idxof((i + mid) % N32)] += r;
				r += init_key[j] + i;
				psfmt32[idxof((i + mid + lag) % N32)] += r;
				psfmt32[idxof(i)] = r;
				i = (i + 1) % N32;
			}
			for (; j < count; j++) {
				r = func1(psfmt32[idxof(i)] ^ psfmt32[idxof((i + mid) % N32)]
					^ psfmt32[idxof((i + N32 - 1) % N32)]);
				psfmt32[idxof((i + mid) % N32)] += r;
				r += i;
				psfmt32[idxof((i + mid + lag) % N32)] += r;
				psfmt32[idxof(i)] = r;
				i = (i + 1) % N32;
			}
			for (j = 0; j < N32; j++) {
				r = func2(psfmt32[idxof(i)] + psfmt32[idxof((i + mid) % N32)]
					+ psfmt32[idxof((i + N32 - 1) % N32)]);
				psfmt32[idxof((i + mid) % N32)] ^= r;
				r -= i;
				psfmt32[idxof((i + mid + lag) % N32)] ^= r;
				psfmt32[idxof(i)] = r;
				i = (i + 1) % N32;
			}
			idx = N32;
			period_certification();
			initialized = 1;
		}
		forceinline void fill_array32(uint32_t *array, int size) {
			assert(initialized);
			assert(idx == N32);
			assert(size % 4 == 0);
			assert(size >= N32);
			gen_rand_array((w128_t *)array, size / 4);
			idx = N32;
		}
		forceinline void fill_array64(uint64_t *array, int size) {
			assert(initialized);
			assert(idx == N32);
			assert(size % 2 == 0);
			assert(size >= N64);
			gen_rand_array((w128_t *)array, size / 2);
			idx = N32;
		}
		forceinline const char *get_idstring(void) const { return idstring; }
		forceinline int get_min_array_size32(void) const { return N32; }
		forceinline int get_min_array_size64(void) const { return N64; }
		forceinline double genrand_real3(void) { return to_real3(gen_rand32()); }
		forceinline double genrand_real2(void) { return to_real2(gen_rand32()); }
		forceinline double genrand_real1(void) { return to_real1(gen_rand32()); }
		forceinline double genrand_res53(void) { return to_res53(gen_rand64()); }
		forceinline double genrand_res53_mix(void) { 
			uint32_t x, y;
			x = gen_rand32();
			y = gen_rand32();
			return to_res53_mix(x, y);
		}
		uint32_t gen_rand32(void) {
			uint32_t r;

			assert(initialized);
			if (idx >= N32) {
				gen_rand_all();
				idx = 0;
			}
			r = psfmt32[idx++];
			return r;
		}
		uint64_t gen_rand64(void) {
			uint64_t r;
			assert(initialized);
			assert(idx % 2 == 0);

			if (idx >= N32) {
				gen_rand_all();
				idx = 0;
			}
			r = psfmt64[idx / 2];
			idx += 2;
			return r;
		}
		//return 128-bit xmm register
		__m128i gen_randx128(void) {
			__m128i r;
			assert(initialized);
			assert(idx % 2 == 0);

			if (idx >= N32) {
				gen_rand_all();
				idx = 0;
			}
			r = sfmt[idx / 4].si;
			idx += 4;
			return r;
		}
	};

	typedef sse2_fast_mersenne_twister<607, 2, 15, 3, 13, 3,
		0xfdff37ffU, 0xef7f3f7dU, 0xff777b7dU, 0x7ff7fb2fU,
		0x00000001U, 0x00000000U, 0x00000000U, 0x5986f054U> sfmt607;

	typedef sse2_fast_mersenne_twister<1279, 7, 14, 3, 5, 1,
		0xf7fefffdU, 0x7fefcfffU, 0xaff3ef3fU, 0xb5ffff7fU,
		0x00000001U, 0x00000000U, 0x00000000U, 0x20000000U> sfmt1279;

	typedef sse2_fast_mersenne_twister<2281, 12, 19, 1, 5, 1,
		0xbff7ffbfU, 0xfdfffffeU, 0xf7ffef7fU, 0xf2f7cbbfU,
		0x00000001U, 0x00000000U, 0x00000000U, 0x41dfa600U> sfmt2281;

	typedef sse2_fast_mersenne_twister<4253, 17, 20, 1, 7, 1,
		0x9f7bffff, 0x9fffff5f, 0x3efffffb, 0xfffff7bb,
		0xa8000001U, 0xaf5390a3U, 0xb740b3f8U, 0x6c11486dU> sfmt4253;

	typedef sse2_fast_mersenne_twister<12213, 68, 14, 3, 7, 3, 
		0xeffff7fbU, 0xffffffefU, 0xdfdfbfffU, 0x7fffdbfdU,
		0x00000001U, 0x00000000U, 0xe8148000U, 0xd0c7afa3U> sfmt12213;

	typedef sse2_fast_mersenne_twister<19937, 122, 18, 1, 11, 1, 
		0xdfffffefU, 0xddfecb7fU, 0xbffaffffU, 0xbffffff6U,
		0x00000001U, 0x00000000U, 0x00000000U, 0x13c9e684U> sfmt19937;

	typedef sse2_fast_mersenne_twister<44497, 330, 5, 3, 9, 3,
		0xeffffffbU, 0xdfbebfffU, 0xbfbf7befU, 0x9ffd7bffU,
		0x00000001U, 0x00000000U, 0xa3ac4000U, 0xecc1327aU> sfmt44497;

	typedef sse2_fast_mersenne_twister<86243, 366, 6, 7,19, 1,
		0xfdbffbffU, 0xbff7ff3fU, 0xfd77efffU, 0xbf9ff3ffU, 
		0x00000001U, 0x00000000U, 0x00000000U, 0xe9528d85U> sfmt86243;

	typedef sse2_fast_mersenne_twister<132049, 110, 19, 1, 21, 1, 
		0xffffbb5fU, 0xfb6ebf95U, 0xfffefffaU, 0xcff77fffU,
		0x00000001U, 0x00000000U, 0xcb520000U, 0xc7e91c7dU> sfmt132049; 

	typedef sse2_fast_mersenne_twister<216091, 627, 11, 3, 10, 1,
		0xbff7bff7U, 0xbfffffffU, 0xbffffa7fU, 0xffddfbfbU, 
		0xf8000001U, 0x89e80709U, 0x3bd2b64bU, 0x0c64b1e4U> sfmt216091;

};//sfmtplus

#endif //SSE2_SFMT_PLUS__