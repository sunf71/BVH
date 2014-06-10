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

/*! \file random.h
 *   \brief Defines a random number generator class.
 */

#ifndef __NIH_RANDOM_H
#define __NIH_RANDOM_H

#include "basic/numbers.h"
#include "linalg/vector.h"
#include <algorithm>

namespace nih {

/*! \addtogroup sampling Sampling
 */

/*! \addtogroup random Random Sequences
 *  \ingroup sampling
 *  \{
 */

#define MTRAND_TYPE		0
#define SFMTRAND_TYPE	1
#define MWCRAND_TYPE	2

//#define RANDOM_TYPE		MTRAND_TYPE
//#define RANDOM_TYPE		MWCRAND_TYPE
#define RANDOM_TYPE		SFMTRAND_TYPE

#if (RANDOM_TYPE == MTRAND_TYPE)

#include <nih/mtrand/mtrand.h>

///
/// Random number generator
///
class Random
{
public:
	/// reset the seed
    ///
    /// \param s    new seed
	void seed(const uint32 s) { m_mt.seed(s); }

	/// return a random [0,1) number
    ///
	float next() { return float( m_mt() ); }

private:
	MTRand_open m_mt;
};

#elif (RANDOM_TYPE == SFMTRAND_TYPE)

#include "sfmtrand/sfmtrand.h"

class Random
{
public:
	Random() { m_mt.init_gen_rand(5489ul); }

	/// seed
	void seed(const uint32 s) { m_mt.init_gen_rand(s); }

	/// return a random [0,1) number
	float next() { return float( m_mt.genrand_real2() ); }

private:
	sfmtplus::sfmt1279 m_mt;
};

#else

#include <nih/mwcrand/mwcrand.h>

///
/// Random number generator
///
class Random
{
public:
	/// reset the seed
    ///
    /// \param s    new seed
	void seed(const uint32 s) { m_mt.seed_random(s); }

	/// return a random [0,1) number
    ///
	float next() { return float( m_mt.get_unit_float() ); }

private:
	mwcrand::generator m_mt;
};

#endif

/*! \}
 */

} // namespace nih

#endif