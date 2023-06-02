// Copyright 2021,2023 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Utility.hpp"
#include "VOLKBlock.hpp"

#include <Pothos/Exception.hpp>

#include <volk/volk.h>

#include <cstring>

//
// Interface
//

class Normalize: public VOLKBlock
{
    public:
        static Pothos::Block* make();

        Normalize();
        virtual ~Normalize() = default;

        void work() override;

        float scalar() const
        {
            return _scalar;
        }

        void setScalar(float scalar)
        {
            _scalar = scalar;
        }

    private:
        float _scalar;
};

//
// Implementation
//

Pothos::Block* Normalize::make()
{
    return new Normalize();
}

Normalize::Normalize(): VOLKBlock(), _scalar(1.0f)
{
    this->setupInput(0, "float32");
    this->setupOutput(0, "float32");

    this->registerCall(this, POTHOS_FCN_TUPLE(Normalize, scalar));
    this->registerCall(this, POTHOS_FCN_TUPLE(Normalize, setScalar));
}

void Normalize::work()
{
    const auto elems = this->workInfo().minElements;
    if(0 == elems) return;

    auto input = this->input(0);
    auto output = this->output(0);

    std::memcpy(output->buffer(), input->buffer(), elems * sizeof(float));
    volk_32f_s32f_normalize(
        output->buffer(),
        _scalar,
        static_cast<unsigned int>(elems));

    input->consume(elems);
    output->produce(elems);
}

/***********************************************************************
 * |PothosDoc Normalize (VOLK)
 *
 * <p>
 * Divides each input by the user-given scalar.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32f_s32f_normalize</b>
 * </p>
 *
 * |category /Math/VOLK
 * |category /VOLK/Math
 * |keywords divide
 *
 * |param scalar[Scalar]
 * A normalization factor to be applied to each input element.
 * |widget DoubleSpinBox(decimals=3)
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/normalize()
 * |setter setScalar(scalar)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKNormalize(
    "/volk/normalize",
    &Normalize::make);
