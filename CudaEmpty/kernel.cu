
typedef unsigned int uint32;



template<typename T>
struct SimpleList
{
	template< typename T >
	T* myCudaMalloc( int size )
	{
		T* loc          = NULL;
		const int space = size * sizeof( T );
		cudaMalloc( &loc, space );
		return loc;
	}
	
	SimpleList(uint32 size):_size(size),_current(0)
	{		
		_data = myCudaMalloc<T>(size);
	}
	__host__ __device__ void add(T value)
	{
		if (_current<_size)
		_data[_current++] = value;
	}
	__host__ __device__ uint32 size()
	{
		return _current;
	}
	__host__ __device__ T* getData()
	{
		return _data;
	}
	__host__ __device__ T operator[](uint32 i)
	{
		if (i<_size)
			return _data[i];
		else
			return NULL;
	}
	T* _data;
	uint32 _current;
	uint32 _size;
};
//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char** argv)
{
	uint32* data;
    SimpleList<uint32> list(10);
    return 0;
}
