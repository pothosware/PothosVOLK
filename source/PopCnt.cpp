// Copyright 2021,2023 Nicholas Corgan
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
    this->setupInput(0, "uint64");
    this->setupOutput(0, "uint64");
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

/***********************************************************************
 * |PothosDoc Population Count (VOLK)
 *
 * <p>
 * For each element, output the population count (or Hamming distance).
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_64u_popcnt</b>
 * </p>
 *
 * |category /Digital/VOLK
 * |category /VOLK/Digital
 * |keywords bit population hamming distance
 *
 * |factory /volk/popcnt()
 **********************************************************************/
static Pothos::BlockRegistry registerPopCnt(
    "/volk/popcnt",
    &PopCnt::make);
