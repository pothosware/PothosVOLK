// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Pothos/Framework.hpp>

//
// OneToOneBlock
//

template <typename InType, typename OutType>
using OneToOneFcn = void(*)(OutType*, const InType*, unsigned int);

template <typename InType, typename OutType>
class OneToOneBlock: public Pothos::Block
{
    public:
        using Class = OneToOneBlock<InType, OutType>;
        using Fcn = OneToOneFcn<InType, OutType>;

        static Pothos::Block* make(Fcn fcn)
        {
            return new Class(fcn);
        }

        OneToOneBlock(Fcn fcn): _fcn(fcn)
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
// OneToOneScalarParamBlock
//

template <typename InType, typename OutType, typename ScalarType>
using OneToOneScalarParamFcn = void(*)(OutType*, const InType*, const ScalarType, unsigned int);

template <typename InType, typename OutType, typename ScalarType>
class OneToOneScalarParamBlock: public Pothos::Block
{
    public:
        using Class = OneToOneScalarParamBlock<InType, OutType, ScalarType>;
        using Fcn = OneToOneScalarParamFcn<InType, OutType, ScalarType>;

        static Pothos::Block* make(Fcn fcn)
        {
            return new Class(fcn);
        }

        OneToOneScalarParamBlock(Fcn fcn): _fcn(fcn)
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

//
// TwoToOneBlock
//

template <typename InType0, typename InType1, typename OutType>
using TwoToOneFcn = void(*)(OutType*, const InType0*, const InType1*, unsigned int);

template <typename InType0, typename InType1, typename OutType>
class TwoToOneBlock: public Pothos::Block
{
    public:
        using Class = TwoToOneBlock<InType0, InType1, OutType>;
        using Fcn = TwoToOneFcn<InType0, InType1, OutType>;

        static Pothos::Block* make(Fcn fcn)
        {
            return new Class(fcn);
        }

        TwoToOneBlock(Fcn fcn): _fcn(fcn)
        {
            this->setupInput(0, Pothos::DType(typeid(InType0)));
            this->setupInput(1, Pothos::DType(typeid(InType1)));
            this->setupOutput(0, Pothos::DType(typeid(OutType)));
        }

        void work() override
        {
            const auto elems = this->workInfo().minElements;
            if(0 == elems) return;

            auto input0 = this->input(0);
            auto input1 = this->input(1);
            auto output = this->output(0);

            _fcn(output->buffer(), input0->buffer(), input1->buffer(), static_cast<unsigned int>(elems));

            input0->consume(elems);
            input1->consume(elems);
            output->produce(elems);
        }

    private:
        Fcn _fcn;
};
