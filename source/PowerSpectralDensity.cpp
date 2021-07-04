// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VOLKBlock.hpp"

#include <volk/volk.h>

//
// Interface
//

class PowerSpectralDensity: public VOLKBlock
{
    public:
        static Pothos::Block* make();

        PowerSpectralDensity();
        virtual ~PowerSpectralDensity() = default;

        float normalizationFactor() const
        {
            return _normalizationFactor;
        }

        void setNormalizationFactor(float normalizationFactor)
        {
            _normalizationFactor = normalizationFactor;
        }

        float rbw() const
        {
            return _rbw;
        }

        void setRBW(float rbw)
        {
            _rbw = rbw;
        }

        void work() override;

    private:
        float _normalizationFactor;
        float _rbw;
};

//
// Implementation
//

Pothos::Block* PowerSpectralDensity::make()
{
    return new PowerSpectralDensity();
}

PowerSpectralDensity::PowerSpectralDensity():
    VOLKBlock(),
    _normalizationFactor(1.0f),
    _rbw(1.0f)
{
    static const Pothos::DType InDType(typeid(std::complex<float>));
    static const Pothos::DType OutDType(typeid(float));

    this->setupInput(0, InDType);
    this->setupOutput(0, OutDType);

    this->registerCall(this, POTHOS_FCN_TUPLE(PowerSpectralDensity, normalizationFactor));
    this->registerCall(this, POTHOS_FCN_TUPLE(PowerSpectralDensity, setNormalizationFactor));

    this->registerCall(this, POTHOS_FCN_TUPLE(PowerSpectralDensity, rbw));
    this->registerCall(this, POTHOS_FCN_TUPLE(PowerSpectralDensity, setRBW));
}

void PowerSpectralDensity::work()
{
    const auto elems = this->workInfo().minElements;
    if(0 == elems) return;

    auto input = this->input(0);
    auto output = this->output(0);

    volk_32fc_s32f_x2_power_spectral_density_32f(
        output->buffer(),
        input->buffer(),
        _normalizationFactor,
        _rbw,
        elems);

    input->consume(elems);
    output->produce(elems);
}

//
// Factory
//

static Pothos::BlockRegistry registerVOLKPowerSpectralDensity(
    "/volk/power_spectral_density",
    &PowerSpectralDensity::make);
