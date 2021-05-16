// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "SharedBufferAllocator.hpp"

#include <Pothos/Framework.hpp>

//
// VOLKBlock
//

class VOLKBlock: public Pothos::Block
{
    public:
        VOLKBlock(){}
        virtual ~VOLKBlock() = default;

        Pothos::BufferManager::Sptr getInputBufferManager(
            const std::string&,
            const std::string&) override
        {
            auto bufferManager = Pothos::BufferManager::make("generic");
            bufferManager->setAllocateFunction(&volkSharedBufferAllocator);

            return bufferManager;
        }

        Pothos::BufferManager::Sptr getOutputBufferManager(
            const std::string&,
            const std::string&) override
        {
            auto bufferManager = Pothos::BufferManager::make("generic");
            bufferManager->setAllocateFunction(&volkSharedBufferAllocator);

            return bufferManager;
        }
};

//
// OneToOneBlock
//

template <typename InType, typename OutType>
using OneToOneFcn = void(*)(OutType*, const InType*, unsigned int);

template <typename InType, typename OutType>
class OneToOneBlock: public VOLKBlock
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
            static const Pothos::DType InDType(typeid(InType));
            static const Pothos::DType OutDType(typeid(OutType));

            this->setupInput(0, InDType);
            this->setupOutput(0, OutDType);
        }

        virtual ~OneToOneBlock() = default;

        void work() override
        {
            const auto elems = this->workInfo().minElements;
            if(0 == elems) return;

            auto input = this->input(0);
            auto output = this->output(0);

            _fcn(output->buffer().template as<OutType*>(),
                 input->buffer().template as<const InType*>(),
                 static_cast<unsigned int>(elems));

            input->consume(elems);
            output->produce(elems);
        }

    protected:
        Fcn _fcn;
};

//
// OneToOneScalarParamBlock
//

template <typename InType, typename OutType, typename ScalarType>
using OneToOneScalarParamFcn = void(*)(OutType*, const InType*, const ScalarType, unsigned int);

template <typename InType, typename OutType, typename ScalarType>
class OneToOneScalarParamBlock: public VOLKBlock
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
            static const Pothos::DType InDType(typeid(InType));
            static const Pothos::DType OutDType(typeid(OutType));

            this->setupInput(0, InDType);
            this->setupOutput(0, OutDType);

            this->registerCall(this, POTHOS_FCN_TUPLE(Class, scalar));
            this->registerCall(this, POTHOS_FCN_TUPLE(Class, setScalar));
        }

        virtual ~OneToOneScalarParamBlock() = default;

        virtual ScalarType scalar() const
        {
            return _scalar;
        }

        virtual void setScalar(ScalarType scalar)
        {
            _scalar = scalar;
        }

        void work() override
        {
            const auto elems = this->workInfo().minElements;
            if(0 == elems) return;

            auto input = this->input(0);
            auto output = this->output(0);

            _fcn(output->buffer().template as<OutType*>(),
                 input->buffer().template as<const InType*>(),
                 _scalar,
                 static_cast<unsigned int>(elems));

            input->consume(elems);
            output->produce(elems);
        }

    protected:
        Fcn _fcn;
        ScalarType _scalar;
};

//
// OneToTwoBlock
//

template <typename InType, typename OutType0, typename OutType1>
using OneToTwoFcn = void(*)(OutType0*, OutType1*, const InType*, unsigned int);

template <typename InType, typename OutType0, typename OutType1>
class OneToTwoBlock: public VOLKBlock
{
    public:
        using Class = OneToTwoBlock<InType, OutType0, OutType1>;
        using Fcn = OneToTwoFcn<InType, OutType0, OutType1>;

        static Pothos::Block* make(Fcn fcn)
        {
            return new Class(fcn);
        }

        OneToTwoBlock(Fcn fcn): _fcn(fcn)
        {
            static const Pothos::DType InDType(typeid(InType));
            static const Pothos::DType OutDType0(typeid(OutType0));
            static const Pothos::DType OutDType1(typeid(OutType1));

            this->setupInput(0, InDType);
            this->setupOutput(0, OutDType0);
            this->setupOutput(1, OutDType1);
        }

        virtual ~OneToTwoBlock() = default;

        void work() override
        {
            const auto elems = this->workInfo().minElements;
            if(0 == elems) return;

            auto input = this->input(0);
            auto output0 = this->output(0);
            auto output1 = this->output(0);

            _fcn(output0->buffer().template as<OutType0*>(),
                 output1->buffer().template as<OutType1*>(),
                 input->buffer().template as<const InType*>(),
                 static_cast<unsigned int>(elems));

            input->consume(elems);
            output0->produce(elems);
            output1->produce(elems);
        }

    protected:
        Fcn _fcn;
};

//
// TwoToOneBlock
//

template <typename InType0, typename InType1, typename OutType>
using TwoToOneFcn = void(*)(OutType*, const InType0*, const InType1*, unsigned int);

template <typename InType0, typename InType1, typename OutType>
class TwoToOneBlock: public VOLKBlock
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
            static const Pothos::DType InDType0(typeid(InType0));
            static const Pothos::DType InDType1(typeid(InType1));
            static const Pothos::DType OutDType(typeid(OutType));

            this->setupInput(0, InDType0);
            this->setupInput(1, InDType1);
            this->setupOutput(0, OutDType);
        }

        virtual ~TwoToOneBlock() = default;

        void work() override
        {
            const auto elems = this->workInfo().minElements;
            if(0 == elems) return;

            auto input0 = this->input(0);
            auto input1 = this->input(1);
            auto output = this->output(0);

            _fcn(output->buffer().template as<OutType*>(),
                 input0->buffer().template as<const InType0*>(),
                 input1->buffer().template as<const InType1*>(),
                 static_cast<unsigned int>(elems));

            input0->consume(elems);
            input1->consume(elems);
            output->produce(elems);
        }

    protected:
        Fcn _fcn;
};
