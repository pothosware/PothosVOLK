// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Utility.hpp"
#include "VOLKBlock.hpp"

#include <Pothos/Exception.hpp>

#include <volk/volk.h>

#include <cstring>

template <typename T>
class Byteswap: public VOLKBlock
{
    public:
        using Fcn = void(*)(T*, unsigned int);

        Byteswap(Fcn fcn):
            VOLKBlock(),
            _byteswapFcn(fcn)
        {
            static const Pothos::DType dtype(typeid(T));

            this->setupInput(0, dtype);
            this->setupOutput(0, dtype);
        };
        virtual ~Byteswap() = default;

        void work() override
        {
            const auto elems = this->workInfo().minElements;
            if(0 == elems) return;

            auto input = this->input(0);
            auto output = this->output(0);

            std::memcpy(output->buffer(), input->buffer(), elems * sizeof(T));
            _byteswapFcn(output->buffer().template as<T*>(), static_cast<unsigned int>(elems));

            input->consume(elems);
            output->produce(elems);
        }

    private:
        Fcn _byteswapFcn;
};

static const std::string VOLKByteswapPath = "/volk/byteswap";

#define IfTypeThenByteswap(Type,fcn) \
    if(doesDTypeMatch<Type>(dtype)) return new Byteswap<Type>(fcn);

static Pothos::Block* makeByteswap(const Pothos::DType& dtype)
{
    IfTypeThenByteswap(uint16_t,volk_16u_byteswap)
    IfTypeThenByteswap(uint32_t,volk_32u_byteswap)
    IfTypeThenByteswap(uint64_t,volk_64u_byteswap)

    throw InvalidDTypeException(VOLKByteswapPath, std::vector<Pothos::DType>{dtype});
}

static Pothos::BlockRegistry registerVOLKByteswap(
    VOLKByteswapPath,
    &makeByteswap);
