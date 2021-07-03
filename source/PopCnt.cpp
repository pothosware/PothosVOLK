// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VOLKBlock.hpp"

#include <volk/volk.h>

//
// Interface
//

class PopCnt: public VOLKBlock
{
    public:
        static Pothos::Block* make();

        PopCnt();
        virtual ~PopCnt() = default;

        void work() override;
};

//
// Implementation
//

Pothos::Block* PopCnt::make()
{
    return new PopCnt();
}

PopCnt::PopCnt(): VOLKBlock()
{
    static const Pothos::DType DType(typeid(uint64_t));

    this->setupInput(0, DType);
    this->setupOutput(0, DType);
}

void PopCnt::work()
{
    const auto elems = this->workInfo().minElements;
    if(0 == elems) return;

    auto input = this->input(0);
    auto output = this->output(0);

    const uint64_t* inputBuffer = input->buffer();
    uint64_t* outputBuffer = output->buffer();

    for(size_t elem = 0; elem < elems; ++elem)
    {
        volk_64u_popcnt(
            &outputBuffer[elem],
            inputBuffer[elem]);
    }

    input->consume(elems);
    output->produce(elems);
}

//
// Factory
//

static Pothos::BlockRegistry registerPopCnt(
    "/volk/popcnt",
    &PopCnt::make);
