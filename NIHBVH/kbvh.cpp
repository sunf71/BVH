#include "bits\morton.h"
#include "kbvh.h"
namespace nih {

	template <uint32 DIM>
template <typename Iterator>
void KBvh_builder<DIM>::build(
	const Bbox3f    bbox,
	const Iterator	begin,
	const Iterator	end,
	Bvh<DIM>*		bvh)
{
	//assign morton codes
	morton_functor<DIM> functor(bbox);
	for(Iterator itr = begin;
		itr != end;
		itr++)
	{
		m_codes.push_back(functor(*Itr));
	}
	//sort morton codes

}
}