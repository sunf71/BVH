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

#include <base/vector.h>
#include <cmath>

namespace nih {

template <typename T> struct Sparse_matrix_entry
{
public:
	inline  Sparse_matrix_entry (void);
	inline  Sparse_matrix_entry (int32 i, int32 j, const T& v);
	inline  Sparse_matrix_entry (const Sparse_matrix_entry& en);
	inline ~Sparse_matrix_entry ();

	inline Sparse_matrix_entry& operator  = (const T& v);
	inline Sparse_matrix_entry& operator  = (const Sparse_matrix_entry& en);
	inline bool   operator == (const Sparse_matrix_entry& en) const;
	inline bool   operator != (const Sparse_matrix_entry& en) const;

	inline int32     row   (void) const;
	inline int32     col   (void) const;
	inline const T& value (void) const;

	inline int32 set (int32 i, int32 j, const T& v);

public:
	int32 dRow;
	int32 dCol;
	T     dVal;
};

template <typename T> struct Sparse_matrix
{
public:
	typedef T value_type;

    typedef Sparse_matrix_entry<T> Entry;

	class const_iterator
	{
	public:
		typedef Sparse_matrix_entry<T> Entry;
		
	public:
		inline  const_iterator (void);
		inline  const_iterator (const Sparse_matrix<T> *m, int32 k);
		inline  const_iterator (const const_iterator& it);
		inline ~const_iterator ();

		inline const_iterator& operator  = (const const_iterator& it);
		inline bool           operator == (const const_iterator& it) const;
		inline bool           operator != (const const_iterator& it) const;

		inline const Entry&   operator* (void) const;

		inline const_iterator& operator++ (void);
		inline const_iterator& operator++ (int);

		inline const_iterator& operator-- (void);
		inline const_iterator& operator-- (int);

		inline const_iterator& operator+= (int32);
		inline const_iterator& operator-= (int32);

		inline const_iterator  operator + (int32) const;
		inline const_iterator  operator - (int32) const;
		inline int32           operator - (const const_iterator&) const;
		inline const Entry&   operator[] (int32) const;

		inline int32     row   (void) const;
		inline int32     col   (void) const;
		inline const T& value (void) const;

	private:
		const Sparse_matrix<T> *dM;
		int32                   dK;
	};

	class iterator
	{
	public:
		typedef Sparse_matrix_entry<T> Entry;

	public:
		inline  iterator (void);
		inline  iterator (Sparse_matrix<T> *m, int32 k);
		inline  iterator (const iterator& it);
		inline ~iterator ();

		inline iterator& operator  = (const iterator& it);
		inline bool      operator == (const iterator& it) const;
		inline bool      operator != (const iterator& it) const;

		inline const Entry& operator* (void) const;
		inline Entry&       operator* (void);

		inline iterator& operator++ (void);
		inline iterator& operator++ (int);

		inline iterator& operator-- (void);
		inline iterator& operator-- (int);

		inline iterator& operator+= (int32);
		inline iterator& operator-= (int32);

		inline iterator  operator + (int32) const;
		inline iterator  operator - (int32) const;
		inline int32      operator - (const iterator&) const;
		inline Entry&    operator[] (int32) const;

		inline int32     row   (void) const;
		inline int32     col   (void) const;
		inline const T& value (void) const;

		inline int32 set (int32 i, int32 j, const T& v);

	private:
		Sparse_matrix<T> *dM;
		int32             dK;
	};

public:
inline         Sparse_matrix  ();
inline         Sparse_matrix  (int32 n, int32 m, int32 size = 0);
inline         Sparse_matrix  (const Sparse_matrix<T>&);
inline         Sparse_matrix  (int32 n, int32 m, const T  *v);
inline         Sparse_matrix  (int32 n, int32 m, const T **v);
inline         Sparse_matrix  (int32 n, int32 m, int32 size, const T *v, const int32 *i, const int32 *j);
inline         Sparse_matrix  (int32 n, int32 m, int32 size, Entry* e, int32 reference = 0);
inline        ~Sparse_matrix  ();

inline        void resize (int32 n, int32 m);
inline        void resize (int32 n, int32 m, int32 size);

inline        int32 nRows (void) const { return dRows; }
inline        int32 nCols (void) const { return dCols; }
inline        int32 size  (void) const { return dSize; }

inline        Sparse_matrix<T>&  operator  = (const Sparse_matrix<T>&);
inline        Sparse_matrix<T>&  operator *= (const T&);
inline        Sparse_matrix<T>&  operator /= (const T&);

inline        Sparse_matrix<T>&  transpose (void);

inline        const_iterator begin (void) const;
inline        const_iterator end   (void) const;
inline        iterator      begin (void);
inline        iterator      end   (void);
inline        const Entry&  operator [] (int32 i) const;
inline        Entry&        operator [] (int32 i);

friend NIH_API_CS int  operator == <T> (const Sparse_matrix<T>&,  const Sparse_matrix<T>&);
friend NIH_API_CS int  operator != <T> (const Sparse_matrix<T>&,  const Sparse_matrix<T>&);

public:
	int32   dRows;
	int32   dCols;
	int32   dSize;
	Entry *dEntries;
	int32   dRef;
};

template<typename T, class RowVector, class ColVector> ColVector& mult  (const RowVector&,         const Sparse_matrix<T>&, ColVector&);
template<typename T, class RowVector, class ColVector> RowVector& mult  (const Sparse_matrix<T>&, const ColVector&,         RowVector&);
template<typename T, class RowVector, class ColVector> ColVector& madd  (const RowVector&,         const Sparse_matrix<T>&, ColVector&);
template<typename T, class RowVector, class ColVector> RowVector& madd  (const Sparse_matrix<T>&, const ColVector&,         RowVector&);
template<typename T> Sparse_matrix<T>&  transpose (const Sparse_matrix<T>&,  Sparse_matrix<T>&);


//
// I M P L E M E N T A T I O N
//

//
// Sparse_matrix inline methods
//

template < typename T > Sparse_matrix<T>::Sparse_matrix () :
	dRows ( 0 ),
	dCols ( 0 ),
	dSize ( 0 ),
	dEntries ( 0 ),
	dRef     ( 0 )
{}

template < typename T > Sparse_matrix<T>::Sparse_matrix  (int32 n, int32 m, int32 size) :
	dRows ( n ),
	dCols ( m ),
	dSize ( size ),
	dEntries ( 0 ),
	dRef     ( 0 )
{
	if (dSize)
		dEntries = new Entry [ dSize ];
}

template < typename T > Sparse_matrix<T>::Sparse_matrix (const Sparse_matrix<T>& m) :
	dRows ( m.dRows ),
	dCols ( m.dCols ),
	dSize ( m.dSize ),
	dEntries ( 0 ),
	dRef     ( 0 )
{
	dEntries = new Entry [ dSize ];

	int32 i;
	for (i = 0; i < dSize; i++)
		dEntries[i] = m.dEntries[i];
//	for (i = 0; i < dRows; i++)
//		dRowPtr[i] = m.dRowPtr[i];
}

template < typename T > Sparse_matrix<T>::Sparse_matrix (int32 n, int32 m, const T *a) :
	dRows ( n ),
	dCols ( m ),
	dSize ( 0 ),
	dEntries ( 0 ),
	dRef     ( 0 )
{
	int32 i, j;

	for (i = 0; i < n * m; i++)
		dSize += a[i] != 0 ? 1 : 0;

	dEntries = new Entry [ dSize ];

	int32 k = 0;
	for (i = 0; i < n; i++)
	{
//		dRowPtr[i] = k;
		for (j = 0; j < m; j++)
		{
			if (a[i*m + j] == 0)
				continue;

			dEntries[k].dVal = a[i*m + j];
			dEntries[k].dRow = i;
			dEntries[k].dCol = j;
			k++;
		}
	}
}

template < typename T > Sparse_matrix<T>::Sparse_matrix (int32 n, int32 m, const T **a) :
	dRows ( n ),
	dCols ( m ),
	dSize ( 0 ),
	dEntries ( 0 ),
	dRef     ( 0 )
{
	int32 i, j;

	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++)
			dSize += a[i][j] != 0 ? 1 : 0;

	dEntries = new Entry [ dSize ];

	int32 k = 0;
	for (i = 0; i < n; i++)
	{
//		dRowPtr[i] = k;
		for (j = 0; j < m; j++)
		{
			if (a[i][j] == 0)
				continue;

			dEntries[k].dVal = a[i][j];
			dEntries[k].dRow = i;
			dEntries[k].dCol = j;
			k++;
		}
	}
}

template < typename T > Sparse_matrix<T>::Sparse_matrix  (int32 n, int32 m, int32 size, const T *val, const int32 *row, const int32 *col) :
	dRows ( n ),
	dCols ( m ),
	dSize ( size ),
	dEntries ( 0 ),
	dRef     ( 0 )
{
	int32 i;

	dEntries = new Entry [ dSize ];

	for (i = 0; i < dSize; i++)
	{
		dEntries[i].dVal = val[i];
		dEntries[i].dRow = row[i];
		dEntries[i].dCol = row[i];

//		dRowPtr[ ind[i]/dCols ]++;
	}
	// Sort dEntries based on dInd
	// ...
}
template < typename T > Sparse_matrix<T>::Sparse_matrix  (int32 n, int32 m, int32 size, Sparse_matrix_entry<T>* entries, int32 ref) :
	dRows ( n ),
	dCols ( m ),
	dSize ( size ),
	dEntries ( 0 ),
	dRef     ( ref )
{
	int32 i;

	if (dRef)
		dEntries = entries;
	else
	{
		dEntries = new Entry [ dSize ];
		
		for (i = 0; i < dSize; i++)
			dEntries[i] = entries[i];
	}
	// Sort dEntries based on dInd
	// ...
}

template < typename T > Sparse_matrix<T>::~Sparse_matrix()
{
	if (! dRef)
		delete [] dEntries;
}

template < typename T > void Sparse_matrix<T>::resize (int32 n, int32 m)
{
	dRows = n;
	dCols = m;
}
template < typename T > void Sparse_matrix<T>::resize (int32 n, int32 m, int32 size)
{
	dRows = n;
	dCols = m;

	if (dSize != size)
	{
		if (dRef)
		{
			dSize = 0;
			dRef  = 0;
			dEntries = 0;
		}

		dSize = size;
		delete [] dEntries;

		dEntries = new Entry [ dSize ];
		dRef = 0;
	}
}

template < typename T > Sparse_matrix<T>& Sparse_matrix<T>::operator  = (const Sparse_matrix<T>& m)
{
	if (dRef)
	{
		dSize = 0;
		dRef  = 0;
		dEntries = 0;
	}

	if (dSize != m.dSize)
	{
		delete [] dEntries;
		dSize = m.dSize;
		dEntries  = new Entry [ dSize ];
	}
/*	if (dRows != m.dRows)
	{
		delete [] dRowPtr;

		dRows = m.dRows;
		dRowPtr = new int32 [ dRows ];
	}*/
	dRows = m.dRows;
	dCols = m.dCols;

	int32 i;
	for (i = 0; i < dSize; i++)
		dEntries[i] = m.dEntries[i];

//	for (i = 0; i < dRows; i++)
//		dRowPtr[i] = m.dRowPtr[i];

	return *this;
}

template < typename T > Sparse_matrix<T>& Sparse_matrix<T>::operator *= (const T& k)
{
	for (int32 i = 0; i < dSize; i++)
		dEntries[i].dVal *= k;

	return *this;
}

template < typename T > Sparse_matrix<T>& Sparse_matrix<T>::operator /= (const T& k)
{
	for (int32 i = 0; i < dSize; i++)
		dEntries[i].dVal /= k;

	return *this;
}

template < typename T > Sparse_matrix<T>& Sparse_matrix<T>::transpose (void)
{
	int32 k, tmp;
	for (k = 0; k < dSize; k++)
	{
		tmp              = dEntries[k].dRow;
		dEntries[k].dRow = dEntries[k].dCol;
		dEntries[k].dCol = tmp;
	}
	tmp = dRows;
	dRows = dCols;
	dCols = tmp;

	return *this;
}

template < typename T > typename Sparse_matrix<T>::const_iterator Sparse_matrix<T>::begin (void) const
{
	return const_iterator( this, 0 );
}
template < typename T > typename Sparse_matrix<T>::const_iterator Sparse_matrix<T>::end (void) const
{
	return const_iterator( this, dSize );
}
template < typename T > typename Sparse_matrix<T>::iterator Sparse_matrix<T>::begin (void)
{
	return iterator( this, 0 );
}
template < typename T > typename Sparse_matrix<T>::iterator Sparse_matrix<T>::end (void)
{
	return iterator( this, dSize );
}
template < typename T > const Sparse_matrix_entry<T>& Sparse_matrix<T>::operator [] (int32 i) const
{
	return dEntries[i];
}
template < typename T > Sparse_matrix_entry<T>& Sparse_matrix<T>::operator [] (int32 i)
{
	return dEntries[i];
}

//
// Sparse_matrix template < typename T > functions (not members)
//

template < typename T > NIH_API_CS int operator == (const Sparse_matrix<T>& a, const Sparse_matrix<T>& b)
{
	int32 sz = a.dSize;
	if (sz != b.dSize || a.dRows != b.dRows || a.dCols != b.dCols)
		return 0;

	int32 i;
	for (i = 0; i < sz; i++)
	{
		if (a.dEntries[i] != b.dEntries[i])
			return 0;
	}
	return 1;
}

template < typename T > NIH_API_CS int operator != (const Sparse_matrix<T>& a, const Sparse_matrix<T>& b)
{
	return !(a == b);
}

template<typename T, class RowVector, class ColVector> ColVector& mult (const RowVector& v, const Sparse_matrix<T>& A, ColVector& r)
{
	int32 k, i, j;

	r.resize( A.dCols );
	for (i = 0; i < A.dCols; i++)
		r[i] = 0.0;

	for (k = 0; k < A.dSize; k++)
	{
		i = A.dEntries[k].dRow;
		j = A.dEntries[k].dCol;

		r[j] += A.dEntries[k].dVal * v[i];
	}
	return r;
}
template<typename T, class RowVector, class ColVector> RowVector& mult (const Sparse_matrix<T>& A, const ColVector& v, RowVector& r)
{
	int32 k, i, j;

	r.resize( A.dRows );
	for (i = 0; i < A.dRows; i++)
		r[i] = 0.0;

	for (k = 0; k < A.dSize; k++)
	{
		i = A.dEntries[k].dRow;
		j = A.dEntries[k].dCol;

		r[i] += A.dEntries[k].dVal * v[j];
	}
	return r;
}
template<typename T, class RowVector, class ColVector> ColVector& madd (const RowVector& v, const Sparse_matrix<T>& A, ColVector& r)
{
	int32 k, i, j;

	r.resize( A.dCols );
	for (k = 0; k < A.dSize; k++)
	{
		i = A.dEntries[k].dRow;
		j = A.dEntries[k].dCol;

		r[j] += A.dEntries[k].dVal * v[i];
	}
	return r;
}
template<typename T, class RowVector, class ColVector> RowVector& madd (const Sparse_matrix<T>& A, const ColVector& v, RowVector& r)
{
	int32 k, i, j;

	r.resize( A.dRows );
	for (k = 0; k < A.dSize; k++)
	{
		i = A.dEntries[k].dRow;
		j = A.dEntries[k].dCol;

		r[i] += A.dEntries[k].dVal * v[j];
	}
	return r;
}
template<typename T> Sparse_matrix<T>& transpose (const Sparse_matrix<T>& A, Sparse_matrix<T>& B)
{
	B = A;
	return B.transpose();
}

template<typename T, class Vector> Vector operator* (const Vector v, const Sparse_matrix<T>& A)
{
    Vector r;
	int32 k, i, j;

	r.resize( A.dCols );
	for (i = 0; i < A.dCols; i++)
		r[i] = 0.0;

	for (k = 0; k < A.dSize; k++)
	{
		i = A.dEntries[k].dRow;
		j = A.dEntries[k].dCol;

		r[j] += A.dEntries[k].dVal * v[i];
	}
	return r;
}
template<typename T, class Vector> Vector operator* (const Sparse_matrix<T>& A, const Vector& v)
{
    Vector r;
	int32 k, i, j;

	r.resize( A.dRows );
	for (i = 0; i < A.dRows; i++)
		r[i] = 0.0;

	for (k = 0; k < A.dSize; k++)
	{
		i = A.dEntries[k].dRow;
		j = A.dEntries[k].dCol;

		r[i] += A.dEntries[k].dVal * v[j];
	}
	return r;
}


template<typename T> Sparse_matrix_entry<T>::Sparse_matrix_entry (void) :
	dRow( 0 ),
	dCol( 0 ),
	dVal( 0 )
{
}
template<typename T> Sparse_matrix_entry<T>::Sparse_matrix_entry (int32 i, int32 j, const T& v) :
	dRow( i ),
	dCol( j ),
	dVal( v )
{
}
template<typename T> Sparse_matrix_entry<T>::Sparse_matrix_entry (const Sparse_matrix_entry<T>& it) :
	dRow( it.dRow ),
	dCol( it.dCol ),
	dVal( it.dVal )
{
}
template<typename T> Sparse_matrix_entry<T>::~Sparse_matrix_entry ()
{
}

template<typename T> Sparse_matrix_entry<T>& Sparse_matrix_entry<T>::operator = (const T& v)
{
	dVal = v;
	return *this;
}
template<typename T> Sparse_matrix_entry<T>& Sparse_matrix_entry<T>::operator = (const Sparse_matrix_entry<T>& it)
{
	dRow = it.dRow;
	dCol = it.dCol;
	dVal = it.dVal;
	return *this;
}
template<typename T> bool Sparse_matrix_entry<T>::operator == (const Sparse_matrix_entry<T>& it) const
{
	return (dRow == it.dRow) && (dCol == it.dCol) && (dVal == it.dVal);
}
template<typename T> bool Sparse_matrix_entry<T>::operator != (const Sparse_matrix_entry<T>& it) const
{
	return (dRow != it.dRow) || (dCol != it.dCol) || (dVal != it.dVal);
}

template<typename T> int32 Sparse_matrix_entry<T>::row (void) const
{
	return dRow;
}
template<typename T> int32 Sparse_matrix_entry<T>::col (void) const
{
	return dCol;
}
template<typename T> const T& Sparse_matrix_entry<T>::value (void) const
{
	return dVal;
}

template<typename T> int32 Sparse_matrix_entry<T>::set (int32 i, int32 j, const T& v)
{
	dRow = i;
	dCol = j;
	dVal = v;
}


template<typename T> Sparse_matrix<T>::iterator::iterator (void) :
	dM( 0 ),
	dK( 0 )
{
}
template<typename T> Sparse_matrix<T>::iterator::iterator (Sparse_matrix<T> *m, int32 k) :
	dM( m ),
	dK( k )
{
}
template<typename T> Sparse_matrix<T>::iterator::iterator (const typename Sparse_matrix<T>::iterator& it) :
	dM( it.dM ),
	dK( it.dK )
{
}
template<typename T> Sparse_matrix<T>::iterator::~iterator ()
{
}

template<typename T> typename Sparse_matrix<T>::iterator& Sparse_matrix<T>::iterator::operator = (const typename Sparse_matrix<T>::iterator& it)
{
	dM = it.dM;
	dK = it.dK;
	return *this;
}
template<typename T> bool Sparse_matrix<T>::iterator::operator == (const typename Sparse_matrix<T>::iterator& it) const
{
	return (dM == it.dM) && (dK == it.dK);
}
template<typename T> bool Sparse_matrix<T>::iterator::operator != (const typename Sparse_matrix<T>::iterator& it) const
{
	return (dM != it.dM) || (dK != it.dK);
}

template<typename T> const Sparse_matrix_entry<T>& Sparse_matrix<T>::iterator::operator* (void) const
{
	return dM->dEntries[dK];
}
template<typename T> Sparse_matrix_entry<T>& Sparse_matrix<T>::iterator::operator* (void)
{
	return dM->dEntries[dK];
}

template<typename T> typename Sparse_matrix<T>::iterator& Sparse_matrix<T>::iterator::operator++ (void)
{
	dK++;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::iterator& Sparse_matrix<T>::iterator::operator++ (int)
{
	dK++;
	return *this;
}

template<typename T> typename Sparse_matrix<T>::iterator& Sparse_matrix<T>::iterator::operator-- (void)
{
	dK--;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::iterator& Sparse_matrix<T>::iterator::operator-- (int)
{
	dK--;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::iterator& Sparse_matrix<T>::iterator::operator+= (int32 n)
{
	dK += n;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::iterator& Sparse_matrix<T>::iterator::operator-= (int32 n)
{
	dK -= n;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::iterator Sparse_matrix<T>::iterator::operator + (int32 n) const
{
	return iterator( dM, dK + n );
}
template<typename T> typename Sparse_matrix<T>::iterator Sparse_matrix<T>::iterator::operator - (int32 n) const
{
	return iterator( dM, dK - n );
}
template<typename T> int32 Sparse_matrix<T>::iterator::operator - (const typename Sparse_matrix<T>::iterator& it) const
{
	return dK - it.dK;
}
template<typename T> Sparse_matrix_entry<T>& Sparse_matrix<T>::iterator::operator [] (int32 n) const
{
	return dM[ dK + n ];
}

template<typename T> int32 Sparse_matrix<T>::iterator::row (void) const
{
	return dM[dK].row();
}
template<typename T> int32 Sparse_matrix<T>::iterator::col (void) const
{
	return dM[dK].col();
}
template<typename T> const T& Sparse_matrix<T>::iterator::value (void) const
{
	return dM[dK].value();
}

template<typename T> int32 Sparse_matrix<T>::iterator::set (int32 i, int32 j, const T& v)
{
	dM[dK].dRow = i;
	dM[dK].dCol = j;
	dM[dK].dVal = v;
}



template<typename T> Sparse_matrix<T>::const_iterator::const_iterator (void) :
	dM( 0 ),
	dK( 0 )
{
}
template<typename T> Sparse_matrix<T>::const_iterator::const_iterator (const Sparse_matrix<T> *m, int32 k) :
	dM( m ),
	dK( k )
{
}
template<typename T> Sparse_matrix<T>::const_iterator::const_iterator (const typename Sparse_matrix<T>::const_iterator& it) :
	dM( it.dM ),
	dK( it.dK )
{
}
template<typename T> Sparse_matrix<T>::const_iterator::~const_iterator ()
{
}

template<typename T> typename Sparse_matrix<T>::const_iterator& Sparse_matrix<T>::const_iterator::operator = (const typename Sparse_matrix<T>::const_iterator& it)
{
	dM = it.dM;
	dK = it.dK;
	return *this;
}
template<typename T> bool Sparse_matrix<T>::const_iterator::operator == (const typename Sparse_matrix<T>::const_iterator& it) const
{
	return (dM == it.dM) && (dK == it.dK);
}
template<typename T> bool Sparse_matrix<T>::const_iterator::operator != (const typename Sparse_matrix<T>::const_iterator& it) const
{
	return (dM != it.dM) || (dK != it.dK);
}

template<typename T> const Sparse_matrix_entry<T>& Sparse_matrix<T>::const_iterator::operator* (void) const
{
	return dM->dEntries[dK];
}

template<typename T> typename Sparse_matrix<T>::const_iterator& Sparse_matrix<T>::const_iterator::operator++ (void)
{
	dK++;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::const_iterator& Sparse_matrix<T>::const_iterator::operator++ (int)
{
	dK++;
	return *this;
}

template<typename T> typename Sparse_matrix<T>::const_iterator& Sparse_matrix<T>::const_iterator::operator-- (void)
{
	dK--;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::const_iterator& Sparse_matrix<T>::const_iterator::operator-- (int)
{
	dK--;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::const_iterator& Sparse_matrix<T>::const_iterator::operator+= (int32 n)
{
	dK += n;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::const_iterator& Sparse_matrix<T>::const_iterator::operator-= (int32 n)
{
	dK -= n;
	return *this;
}
template<typename T> typename Sparse_matrix<T>::const_iterator Sparse_matrix<T>::const_iterator::operator + (int32 n) const
{
	return const_iterator( dM, dK + n );
}
template<typename T> typename Sparse_matrix<T>::const_iterator Sparse_matrix<T>::const_iterator::operator - (int32 n) const
{
	return const_iterator( dM, dK - n );
}
template<typename T> int32 Sparse_matrix<T>::const_iterator::operator - (const typename Sparse_matrix<T>::const_iterator& it) const
{
	return dK - it.dK;
}
template<typename T> const Sparse_matrix_entry<T>& Sparse_matrix<T>::const_iterator::operator [] (int32 n) const
{
	return dM[ dK + n ];
}

template<typename T> int32 Sparse_matrix<T>::const_iterator::row (void) const
{
	return dM[dK].row();
}
template<typename T> int32 Sparse_matrix<T>::const_iterator::col (void) const
{
	return dM[dK].col();
}
template<typename T> const T& Sparse_matrix<T>::const_iterator::value (void) const
{
	return dM[dK].value();
}

} // namespace nih