// Copyright 2021,2023 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Fallback.hpp"
#include "Utility.hpp"
#include "VOLKBlock.hpp"

#include <volk/volk.h>

#include <complex>

//
// Block
//

template <typename T>
class Accumulator: public VOLKBlock
{
    public:
        using Class = Accumulator<T>;
        using Fcn = OneToOneFcn<T,T>;

        static Pothos::Block* make(Fcn fcn)
        {
            return new Class(fcn);
        }

        Accumulator(Fcn fcn):
            _fcn(fcn),
            _accum(0)
        {
            static const Pothos::DType dtype(typeid(T));

            this->setupInput(0, dtype);
            this->setupOutput(0, dtype, this->uid()); // Unique domain because of buffer forwarding

            this->registerCall(this, POTHOS_FCN_TUPLE(Class, currentSum));
            this->registerCall(this, POTHOS_FCN_TUPLE(Class, reset));

            this->registerProbe("currentSum");
        }

        T currentSum() const
        {
            return _accum;
        }

        void reset()
        {
            _accum = 0;
        }

        void work() override
        {
            auto input = this->input(0);
            const auto elems = input->elements();
            if(elems == 0) return;

            auto output = this->output(0);
            auto buffer = input->takeBuffer();

            T bufferAccum = 0;
            _fcn(&bufferAccum, buffer, static_cast<unsigned int>(elems));
            _accum += bufferAccum;

            input->consume(elems);
            output->postBuffer(std::move(buffer));
        }

    private:
        Fcn _fcn;
        T _accum;
};

/***********************************************************************
 * |PothosDoc Accumulator (VOLK)
 *
 * <p>
 * Stores the total sum of all inputs and forwards the buffer without
 * copying. The overall sum can be probed with <b>currentSum</b>.
 * </p>
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_32f_accumulator_32f</b></li>
 * <li><b>volk_32fc_accumulator_32fc</b></li>
 * </ul>
 *
 * |category /Stream/Stream
 * |category /VOLK/Stream
 *
 * |param dtype[Data Type]
 * |widget DTypeChooser(float32=1,cfloat32=1)
 * |default "float32"
 * |preview disable
 *
 * |factory /volk/accumulator(dtype)
 **********************************************************************/
static const std::string VOLKAccumulatorPath = "/volk/accumulator";

static Pothos::Block* makeAccumulator(const Pothos::DType& dtype)
{
    #define IfTypeThenAccumulator(type,fcn) \
        if(doesDTypeMatch<type>(dtype)) return Accumulator<type>::make(fcn);

    IfTypeThenAccumulator(float,volk_32f_accumulator_s32f)
    IfTypeThenAccumulator(std::complex<float>,volk_32fc_accumulator_s32fc)

    throw InvalidDTypeException(VOLKAccumulatorPath, dtype);
}

static Pothos::BlockRegistry registerVOLKAccumulator(
    VOLKAccumulatorPath,
    &makeAccumulator);
