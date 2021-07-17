// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VOLKBlock.hpp"

#include <volk/volk.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <functional>

//
// Interface
//

class SquareDist: public VOLKBlock
{
    public:
        static Pothos::Block* make();

        SquareDist();
        virtual ~SquareDist() = default;

        inline std::complex<float> complexInput() const
        {
            return _input;
        }

        inline void setComplexInput(const std::complex<float>& input)
        {
            _input = input;
        }

        inline float scalar() const
        {
            return _scalar;
        }

        void setScalar(float scalar);

        void work() override;

    private:
        std::complex<float> _input;
        float _scalar;

        // This reduces branching in the work function and allows the
        // VOLK calls to potentially be inlined, which wouldn't work
        // in a capturing lambda.
        using WorkFcn = void(SquareDist::*)(void);
        WorkFcn _work;

        void _workNoScalar();
        void _workScalar();
};

//
// Implementation
//

Pothos::Block* SquareDist::make()
{
    return new SquareDist();
}

SquareDist::SquareDist():
    VOLKBlock(),
    _scalar(1.0f),
    _work(nullptr)
{
    this->setupInput(0, "complex_float32");
    this->setupOutput(0, "float32");

    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDist, complexInput));
    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDist, setComplexInput));

    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDist, scalar));
    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDist, setScalar));

    // Explicitly call to set the work function.
    this->setScalar(1.0f);
}

void SquareDist::setScalar(float scalar)
{
    _scalar = scalar;

    constexpr float epsilon = 1e-6;
    if(std::abs(_scalar-1.0f) <= epsilon) this->_work = &SquareDist::_workNoScalar;
    else                                  this->_work = &SquareDist::_workScalar;
}

void SquareDist::work()
{
    assert(_work);
    std::mem_fn(_work)(this);
}

void SquareDist::_workNoScalar()
{
    const auto elems = this->workInfo().minElements;
    if(0 == elems) return;

    auto input = this->input(0);
    auto output = this->output(0);

    volk_32fc_x2_square_dist_32f(
        output->buffer(),
        &_input,
        input->buffer(),
        elems);

    input->consume(elems);
    output->produce(elems);
}

void SquareDist::_workScalar()
{
    const auto elems = this->workInfo().minElements;
    if(0 == elems) return;

    auto input = this->input(0);
    auto output = this->output(0);

    volk_32fc_x2_s32f_square_dist_scalar_mult_32f(
        output->buffer(),
        &_input,
        input->buffer(),
        _scalar,
        elems);

    input->consume(elems);
    output->produce(elems);
}

/***********************************************************************
 * |PothosDoc Square Distance (VOLK)
 *
 * <p>
 * Calculates the square distance between a single complex input for
 * each point in a complex vector. Optionally scales the output by a
 * given scalar value.
 * </p>
 *
 * <ul>
 * <li><b>volk_32fc_x2_square_dist_32f</b></li>
 * <li><b>volk_32fc_x2_s32f_square_dist_scalar_mult_32f</b></li>
 * </ul>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math complex
 *
 * |param complexInput[Complex Input]
 * |widget LineEdit()
 * |default 1+0i
 * |preview enable
 *
 * |param scalar[Scalar]
 * A value multiplied by each square distance to form the final output.
 * |widget DoubleSpinBox(decimals=3)
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/square_dist()
 * |setter setComplexInput(complexInput)
 * |setter setScalar(scalar)
 **********************************************************************/
static Pothos::BlockRegistry registerSquareDist(
    "/volk/square_dist",
    &SquareDist::make);
