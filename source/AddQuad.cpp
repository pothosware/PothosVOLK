// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VOLKBlock.hpp"

#include <volk/volk.h>

#include <complex>

//
// Interface
//

class AddQuad: public VOLKBlock
{
    public:
        static Pothos::Block* make();

        AddQuad();
        virtual ~AddQuad() = default;

        void work() override;
};

//
// Implementation
//

Pothos::Block* AddQuad::make()
{
    return new AddQuad();
}

AddQuad::AddQuad(): VOLKBlock()
{
    for(size_t i = 0; i < 5; ++i) this->setupInput(i, "int16");
    for(size_t i = 0; i < 4; ++i) this->setupOutput(i, "int16");
}

void AddQuad::work()
{
    const auto elems = this->workInfo().minElements;
    if(0 == elems) return;

    const auto& inputs = this->inputs();
    const auto& outputs = this->outputs();

    volk_16i_x5_add_quad_16i_x4(
        outputs[0]->buffer(),
        outputs[1]->buffer(),
        outputs[2]->buffer(),
        outputs[3]->buffer(),
        inputs[0]->buffer(),
        inputs[1]->buffer(),
        inputs[2]->buffer(),
        inputs[3]->buffer(),
        inputs[4]->buffer(),
        elems);

    for(auto* input: inputs)   input->consume(elems);
    for(auto* output: outputs) output->produce(elems);
}

/***********************************************************************
 * |PothosDoc Add Quad (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_16i_x5_add_quad_16i_x4</b>
 * </p>
 *
 * |category /VOLK
 *
 * |factory /volk/add_quad()
 **********************************************************************/
static Pothos::BlockRegistry registerAddQuad(
    "/volk/add_quad",
    &AddQuad::make);
