// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VOLKBlock.hpp"

#include <volk/volk.h>

#include <complex>

//
// Interface
//

class SquareDist: public VOLKBlock
{
    public:
        using InType = std::complex<float>;
        using OutType = float;

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

        void work() override;

    private:
        std::complex<float> _input;
};

//
// Implementation
//

Pothos::Block* SquareDist::make()
{
    return new SquareDist();
}

SquareDist::SquareDist(): VOLKBlock()
{
    static const Pothos::DType InDType(typeid(InType));
    static const Pothos::DType OutDType(typeid(OutType));

    this->setupInput(0, InDType);
    this->setupOutput(0, OutDType);

    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDist, complexInput));
    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDist, setComplexInput));
}

void SquareDist::work()
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

/***********************************************************************
 * |PothosDoc Square Distance (VOLK)
 *
 * <p>
 * Calculates the square distance between a single complex input for
 * each point in a complex vector.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32fc_x2_square_dist_32f</b>
 * </p>
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
 * |factory /volk/square_dist()
 * |setter setComplexInput(complexInput)
 **********************************************************************/
static Pothos::BlockRegistry registerSquareDist(
    "/volk/square_dist",
    &SquareDist::make);
