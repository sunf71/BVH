#pragma once

#include <nih/basic/types.h>

namespace nih {

template <uint32 WORDS>
struct Bitmask
{
    NIH_HOST_DEVICE Bitmask() { for (int32 i = 0; i < WORDS; ++i) bits[i] = 0; }

    NIH_HOST_DEVICE void set(const int32 i)
    {
        const int32 word_index = i >> 5;
        const int32 bit_index  = i & 31;
        bits[ word_index ] |= (1 << bit_index);
    }
    NIH_HOST_DEVICE void clear(const int32 i)
    {
        const int32 word_index = i >> 5;
        const int32 bit_index  = i & 31;
        bits[ word_index ] &= ~(1 << bit_index);
    }
    NIH_HOST_DEVICE bool get(const int32 i) const
    {
        const int32 word_index = i >> 5;
        const int32 bit_index  = i & 31;
        return bits[ word_index ] & (1 << bit_index);
    }

    NIH_HOST_DEVICE int32 get_word(const int32 i) const { return bits[i]; }

    int32 bits[WORDS];
};
template <>
struct Bitmask<1>
{
    NIH_HOST_DEVICE Bitmask() : bits(0) {}

    NIH_HOST_DEVICE void set(const int32 i)       { bits |= (1 << i); }
    NIH_HOST_DEVICE void clear(const int32 i)     { bits &= ~(1 << i); }
    NIH_HOST_DEVICE bool get(const int32 i) const { return bits & (1 << i); }

    NIH_HOST_DEVICE int32 get_word(const int32 i) const { return bits; }
    NIH_HOST_DEVICE int32 get_bits(const int32 offset) const { return bits >> offset; }

    int32 bits;
};
template <>
struct Bitmask<2>
{
    NIH_HOST_DEVICE Bitmask() { bits0 = bits1 = 0; }

    NIH_HOST_DEVICE void set(const int32 i)
    {
        const int32 word_index = i >> 5;
        const int32 bit_index  = i & 31;
        if (word_index == 0) bits0 |= (1 << bit_index);
        if (word_index != 0) bits1 |= (1 << bit_index);
    }
    NIH_HOST_DEVICE void clear(const int32 i)
    {
        const int32 word_index = i >> 5;
        const int32 bit_index  = i & 31;
        if (word_index == 0) bits0 &= ~(1 << bit_index);
        if (word_index != 0) bits1 &= ~(1 << bit_index);
    }
    NIH_HOST_DEVICE bool get(const int32 i) const
    {
        const int32 word_index = i >> 5;
        const int32 bit_index  = i & 31;
        return (word_index ? bits1 : bits0) & (1 << bit_index);
    }

    NIH_HOST_DEVICE int32 get_word(const int32 i) const { return i ? bits1 : bits0; }

    NIH_HOST_DEVICE int32 get_bits(const int32 offset) const
    {
        const int32 word_index = offset >> 5;
        const int32 bit_offset = offset & 31;
        return (word_index ? bits1 : bits0) >> bit_offset;
    }

    int32 bits0;
    int32 bits1;
};

} // namespace nih
