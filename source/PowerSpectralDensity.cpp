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
    this->setupInput(0, "complex_float32");
    this->setupOutput(0, "float32");

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

/***********************************************************************
 * |PothosDoc Power Spectral Density (VOLK)
 *
 * <p>
 * Calculates the log10 power value divided by the RBW for each input
 * point.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32fc_s32f_x2_power_spectral_density_32f</b>
 * </p>
 *
 * |category /Math
 * |category /FFT
 * |category /VOLK
 * |keywords math rf
 *
 * |param normalizationFactor[Normalization Factor]
 * Divided against all input values before the power is calculated.
 * |widget DoubleSpinBox(decimals=3)
 * |default 1.0
 * |preview enable
 *
 * |param rbw[RBW]
 * Resolution Bandwidth
 * |widget DoubleSpinBox(decimals=3)
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/power_spectral_density()
 * |setter setNormalizationFactor(normalizationFactor)
 * |setter setRBW(rbw)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKPowerSpectralDensity(
    "/volk/power_spectral_density",
    &PowerSpectralDensity::make);
