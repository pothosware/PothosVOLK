// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Pothos/Framework.hpp>

//
// VOLKBlock
//

template <typename InType, typename OutType>
using VOLKFcn = void(*)(OutType*, const InType*, unsigned int);

template <typename InType, typename OutType>
class VOLKBlock: public Pothos::Block
{
    public:
        using Class = VOLKBlock<InType, OutType>;
        using Fcn = VOLKFcn<InType, OutType>;

        static Pothos::Block* make(Fcn fcn)
        {
            return new VOLKBlock(fcn);
        }

        VOLKBlock(Fcn fcn): _fcn(fcn)
        {
            this->setupInput(0, Pothos::DType(typeid(InType)));
            this->setupOutput(0, Pothos::DType(typeid(OutType)));
        }

        void work() override
        {
            const auto elems = this->workInfo().minElements;
            if(0 == elems) return;

            auto input = this->input(0);
            auto output = this->output(0);

            _fcn(output->buffer(), input->buffer(), static_cast<unsigned int>(elems));

            input->consume(elems);
            output->produce(elems);
        }

    private:
        Fcn _fcn;
};

//
// VOLKScalarParamBlock
//

template <typename InType, typename OutType, typename ScalarType>
using VOLKScalarParamFcn = void(*)(OutType*, const InType*, const ScalarType, unsigned int);

template <typename InType, typename OutType, typename ScalarType>
class VOLKScalarParamBlock: public Pothos::Block
{
    public:
        using Class = VOLKScalarParamBlock<InType, OutType, ScalarType>;
        using Fcn = VOLKScalarParamFcn<InType, OutType, ScalarType>;

        static Pothos::Block* make(Fcn fcn)
        {
            return new VOLKScalarParamBlock(fcn);
        }

        VOLKScalarParamBlock(Fcn fcn): _fcn(fcn)
        {
            this->setupInput(0, Pothos::DType(typeid(InType)));
            this->setupOutput(0, Pothos::DType(typeid(OutType)));

            this->registerCall(this, POTHOS_FCN_TUPLE(Class, scalar));
            this->registerCall(this, POTHOS_FCN_TUPLE(Class, setScalar));
        }

        ScalarType scalar() const
        {
            return _scalar;
        }

        void setScalar(ScalarType scalar)
        {
            _scalar = scalar;
        }

        void work() override
        {
            const auto elems = this->workInfo().minElements;
            if(0 == elems) return;

            auto input = this->input(0);
            auto output = this->output(0);

            _fcn(output->buffer(), input->buffer(), _scalar, static_cast<unsigned int>(elems));

            input->consume(elems);
            output->produce(elems);
        }

    private:
        Fcn _fcn;
        ScalarType _scalar;
};
