// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VOLKBlock.hpp"

#include <volk/volk.h>

#include <complex>

//
// Interface
//

class ModRange: public VOLKBlock
{
    public:
        static Pothos::Block* make();

        ModRange();
        virtual ~ModRange() = default;

        float lowerBound() const
        {
            return _lowerBound;
        }

        void setLowerBound(float lowerBound)
        {
            _lowerBound = lowerBound;
        }

        float upperBound() const
        {
            return _upperBound;
        }

        void setUpperBound(float upperBound)
        {
            _upperBound = upperBound;
        }

        void work() override;

    private:
        float _lowerBound;
        float _upperBound;
};

//
// Implementation
//

Pothos::Block* ModRange::make()
{
    return new ModRange();
}

ModRange::ModRange():
    VOLKBlock(),
    _lowerBound(0),
    _upperBound(0)
{
    this->setupInput(0, "float");
    this->setupOutput(0, "float");

    this->registerCall(this, POTHOS_FCN_TUPLE(ModRange, lowerBound));
    this->registerCall(this, POTHOS_FCN_TUPLE(ModRange, setLowerBound));

    this->registerCall(this, POTHOS_FCN_TUPLE(ModRange, upperBound));
    this->registerCall(this, POTHOS_FCN_TUPLE(ModRange, setUpperBound));
}

void ModRange::work()
{
    const auto elems = this->workInfo().minElements;
    if(0 == elems) return;

    auto input = this->input(0);
    auto output = this->output(0);

    volk_32f_s32f_s32f_mod_range_32f(
        output->buffer(),
        input->buffer(),
        _lowerBound,
        _upperBound,
        elems);

    input->consume(elems);
    output->produce(elems);
}

/***********************************************************************
 * |PothosDoc Mod Range (VOLK)
 *
 * <p>
 * Wraps floating-point numbers to stay within a defined [min,max] range.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32f_s32f_s32f_mod_range_32f</b>
 * </p>
 *
 * |category /Stream
 * |category /VOLK
 * |keywords clamp bound wrap
 *
 * |param lowerBound[Lower Bound]
 * |widget DoubleSpinBox(decimals=3)
 * |default 0.0
 * |preview enable
 *
 * |param upperBound[Upper Bound]
 * |widget DoubleSpinBox(decimals=3)
 * |default 0.0
 * |preview enable
 *
 * |factory /volk/mod_range()
 * |setter setLowerBound(lowerBound)
 * |setter setUpperBound(upperBound)
 **********************************************************************/
static Pothos::BlockRegistry registerModRange(
    "/volk/mod_range",
    &ModRange::make);
