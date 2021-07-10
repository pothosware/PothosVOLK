// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VOLKBlock.hpp"

#include <volk/volk.h>

#include <complex>

//
// Interface
//

class SquareDistScalarMult: public VOLKBlock
{
    public:
        using InType = std::complex<float>;
        using OutType = float;

        static Pothos::Block* make();

        SquareDistScalarMult();
        virtual ~SquareDistScalarMult() = default;

        inline std::complex<float> complexInput() const
        {
            return _input;
        }

        inline void setComplexInput(const std::complex<float>& input)
        {
            _input = input;
        }

        float scalar() const
        {
            return _scalar;
        }

        void setScalar(float scalar)
        {
            _scalar = scalar;
        }

        void work() override;

    private:
        std::complex<float> _input;
        float _scalar;
};

//
// Implementation
//

Pothos::Block* SquareDistScalarMult::make()
{
    return new SquareDistScalarMult();
}

SquareDistScalarMult::SquareDistScalarMult(): VOLKBlock()
{
    static const Pothos::DType InDType(typeid(InType));
    static const Pothos::DType OutDType(typeid(OutType));

    this->setupInput(0, InDType);
    this->setupOutput(0, OutDType);

    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDistScalarMult, complexInput));
    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDistScalarMult, setComplexInput));
    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDistScalarMult, scalar));
    this->registerCall(this, POTHOS_FCN_TUPLE(SquareDistScalarMult, setScalar));
}

void SquareDistScalarMult::work()
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
 * |PothosDoc Square Distance (Scaled) (VOLK)
 *
 * <p>
 * Calculates the square distance between a single complex input for
 * each point in a complex vector scaled by a given scalar value.
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
 * |param scalar[Scalar]
 * A value multiplied by each square distance to form the final output.
 * |widget DoubleSpinBox(decimals=3)
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/square_dist_scalar_mult()
 * |setter setComplexInput(complexInput)
 * |setter setScalar(scalar)
 **********************************************************************/
static Pothos::BlockRegistry registerSquareDistScalarMult(
    "/volk/square_dist_scalar_mult",
    &SquareDistScalarMult::make);
