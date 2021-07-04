// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VOLKBlock.hpp"

#include <volk/volk.h>

#include <complex>

//
// Interface
//

class QuadMaxStar: public VOLKBlock
{
    public:
        static Pothos::Block* make();

        QuadMaxStar();
        virtual ~QuadMaxStar() = default;

        void work() override;
};

//
// Implementation
//

Pothos::Block* QuadMaxStar::make()
{
    return new QuadMaxStar();
}

QuadMaxStar::QuadMaxStar(): VOLKBlock()
{
    static const Pothos::DType DType(typeid(int16_t));

    this->setupInput(0, DType);
    this->setupInput(1, DType);
    this->setupInput(2, DType);
    this->setupInput(3, DType);
    this->setupOutput(0, DType);
}

void QuadMaxStar::work()
{
    const auto elems = this->workInfo().minElements;
    if(0 == elems) return;

    const auto& inputs = this->inputs();
    auto output = this->output(0);

    volk_16i_x4_quad_max_star_16i(
        output->buffer(),
        inputs[0]->buffer(),
        inputs[1]->buffer(),
        inputs[2]->buffer(),
        inputs[3]->buffer(),
        elems);

    for(auto* input: inputs) input->consume(elems);
    output->produce(elems);
}

//
// Factory
//

static Pothos::BlockRegistry registerQuadMaxStar(
    "/volk/quad_max_star",
    &QuadMaxStar::make);
