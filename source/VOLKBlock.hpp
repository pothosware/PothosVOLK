// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "SharedBufferAllocator.hpp"

#include <Pothos/Framework.hpp>

#include <string>

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

        virtual void work() override = 0;
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
            static const Pothos::DType inDType(typeid(InType));
            static const Pothos::DType outDType(typeid(OutType));

            this->setupInput(0, inDType);
            this->setupOutput(0, outDType);
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

        static Pothos::Block* make(
            Fcn fcn,
            const std::string& getterName,
            const std::string& setterName)
        {
            return new Class(fcn, getterName, setterName);
        }

        OneToOneScalarParamBlock(
            Fcn fcn,
            const std::string& getterName,
            const std::string& setterName
        ):
            _fcn(fcn)
        {
            static const Pothos::DType inDType(typeid(InType));
            static const Pothos::DType outDType(typeid(OutType));

            this->setupInput(0, inDType);
            this->setupOutput(0, outDType);

            this->registerCall(this, getterName, &Class::scalar);
            this->registerCall(this, setterName, &Class::setScalar);
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

template <typename InType, typename OutType0, typename OutType1, typename OutputPortType>
class OneToTwoBlock: public VOLKBlock
{
    public:
        using Class = OneToTwoBlock<InType, OutType0, OutType1, OutputPortType>;
        using Fcn = OneToTwoFcn<InType, OutType0, OutType1>;

        static Pothos::Block* make(
            Fcn fcn,
            const OutputPortType& outputPort0Name,
            const OutputPortType& outputPort1Name)
        {
            return new Class(fcn, outputPort0Name, outputPort1Name);
        }

        OneToTwoBlock(
            Fcn fcn,
            const OutputPortType& outputPort0Name,
            const OutputPortType& outputPort1Name
        ):
            _fcn(fcn),
            _outputPort0Name(outputPort0Name),
            _outputPort1Name(outputPort1Name)
        {
            static const Pothos::DType inDType(typeid(InType));
            static const Pothos::DType outDType0(typeid(OutType0));
            static const Pothos::DType outDType1(typeid(OutType1));

            this->setupInput(0, inDType);
            this->setupOutput(_outputPort0Name, outDType0);
            this->setupOutput(_outputPort1Name, outDType1);
        }

        virtual ~OneToTwoBlock() = default;

        void work() override
        {
            const auto elems = this->workInfo().minAllElements;
            if(0 == elems) return;

            auto input = this->input(0);
            auto output0 = this->output(_outputPort0Name);
            auto output1 = this->output(_outputPort1Name);

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
        OutputPortType _outputPort0Name;
        OutputPortType _outputPort1Name;
};

//
// OneToTwoScalarParamBlock
//

template <typename InType, typename OutType0, typename OutType1, typename ScalarType>
using OneToTwoScalarParamFcn = void(*)(OutType0*, OutType1*, const InType*, const ScalarType, unsigned int);

template <typename InType, typename OutType0, typename OutType1, typename ScalarType, typename OutputPortType>
class OneToTwoScalarParamBlock: public VOLKBlock
{
    public:
        using Class = OneToTwoScalarParamBlock<InType, OutType0, OutType1, ScalarType, OutputPortType>;
        using Fcn = OneToTwoScalarParamFcn<InType, OutType0, OutType1, ScalarType>;

        static Pothos::Block* make(
            Fcn fcn,
            const std::string& getterName,
            const std::string& setterName,
            const OutputPortType& outputPort0Name,
            const OutputPortType& outputPort1Name)
        {
            return new Class(fcn, getterName, setterName, outputPort0Name, outputPort1Name);
        }

        OneToTwoScalarParamBlock(
            Fcn fcn,
            const std::string& getterName,
            const std::string& setterName,
            const OutputPortType& outputPort0Name,
            const OutputPortType& outputPort1Name
        ):
            _fcn(fcn),
            _outputPort0Name(outputPort0Name),
            _outputPort1Name(outputPort1Name)
        {
            static const Pothos::DType inDType(typeid(InType));
            static const Pothos::DType outDType0(typeid(OutType0));
            static const Pothos::DType outDType1(typeid(OutType1));
            static const Pothos::DType ScalarDType(typeid(ScalarType));

            this->setupInput(0, inDType);
            this->setupOutput(_outputPort0Name, outDType0);
            this->setupOutput(_outputPort1Name, outDType1);

            this->registerCall(this, getterName, &Class::scalar);
            this->registerCall(this, setterName, &Class::setScalar);
        }

        virtual ~OneToTwoScalarParamBlock() = default;

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
            auto output0 = this->output(_outputPort0Name);
            auto output1 = this->output(_outputPort1Name);

            _fcn(output0->buffer().template as<OutType0*>(),
                 output1->buffer().template as<OutType1*>(),
                 input->buffer().template as<const InType*>(),
                 _scalar,
                 static_cast<unsigned int>(elems));

            input->consume(elems);
            output0->produce(elems);
            output1->produce(elems);
        }

    protected:
        Fcn _fcn;
        ScalarType _scalar;
        OutputPortType _outputPort0Name;
        OutputPortType _outputPort1Name;
};

//
// TwoToOneBlock
//

template <typename InType0, typename InType1, typename OutType>
using TwoToOneFcn = void(*)(OutType*, const InType0*, const InType1*, unsigned int);

template <typename InType0, typename InType1, typename OutType, typename InputPortType>
class TwoToOneBlock: public VOLKBlock
{
    public:
        using Class = TwoToOneBlock<InType0, InType1, OutType, InputPortType>;
        using Fcn = TwoToOneFcn<InType0, InType1, OutType>;

        static Pothos::Block* make(
            Fcn fcn,
            const InputPortType& inputPort0Name,
            const InputPortType& inputPort1Name)
        {
            return new Class(fcn, inputPort0Name, inputPort1Name);
        }

        TwoToOneBlock(
            Fcn fcn,
            const InputPortType& inputPort0Name,
            const InputPortType& inputPort1Name
        ):
            _fcn(fcn),
            _inputPort0Name(inputPort0Name),
            _inputPort1Name(inputPort1Name)
        {
            static const Pothos::DType inDType0(typeid(InType0));
            static const Pothos::DType inDType1(typeid(InType1));
            static const Pothos::DType outDType(typeid(OutType));

            this->setupInput(_inputPort0Name, inDType0);
            this->setupInput(_inputPort1Name, inDType1);
            this->setupOutput(0, outDType);
        }

        virtual ~TwoToOneBlock() = default;

        void work() override
        {
            const auto elems = this->workInfo().minAllElements;
            if(0 == elems) return;

            auto input0 = this->input(_inputPort0Name);
            auto input1 = this->input(_inputPort1Name);
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
        InputPortType _inputPort0Name;
        InputPortType _inputPort1Name;
};

//
// TwoToOneScalarParamBlock
//

template <typename InType0, typename InType1, typename OutType, typename ScalarType>
using TwoToOneScalarParamFcn = void(*)(OutType*, const InType0*, const InType1*, const ScalarType, unsigned int);

template <typename InType0, typename InType1, typename OutType, typename ScalarType, typename InputPortType>
class TwoToOneScalarParamBlock: public VOLKBlock
{
    public:
        using Class = TwoToOneScalarParamBlock<InType0, InType1, OutType, ScalarType, InputPortType>;
        using Fcn = TwoToOneScalarParamFcn<InType0, InType1, OutType, ScalarType>;

        static Pothos::Block* make(
            Fcn fcn,
            const std::string& getterName,
            const std::string& setterName,
            const InputPortType& inputPort0Name,
            const InputPortType& inputPort1Name)
        {
            return new Class(fcn, getterName, setterName, inputPort0Name, inputPort1Name);
        }

        TwoToOneScalarParamBlock(
            Fcn fcn,
            const std::string& getterName,
            const std::string& setterName,
            const InputPortType& inputPort0Name,
            const InputPortType& inputPort1Name
        ):
            _fcn(fcn),
            _inputPort0Name(inputPort0Name),
            _inputPort1Name(inputPort1Name)
        {
            static const Pothos::DType inDType0(typeid(InType0));
            static const Pothos::DType inDType1(typeid(InType1));
            static const Pothos::DType outDType(typeid(OutType));

            this->setupInput(inputPort0Name, inDType0);
            this->setupInput(inputPort1Name, inDType1);
            this->setupOutput(0, outDType);

            this->registerCall(this, getterName, &Class::scalar);
            this->registerCall(this, setterName, &Class::setScalar);
        }

        virtual ~TwoToOneScalarParamBlock() = default;

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
            const auto elems = this->workInfo().minAllElements;
            if(0 == elems) return;

            auto input0 = this->input(_inputPort0Name);
            auto input1 = this->input(_inputPort1Name);
            auto output = this->output(0);

            _fcn(output->buffer().template as<OutType*>(),
                 input0->buffer().template as<const InType0*>(),
                 input1->buffer().template as<const InType1*>(),
                 _scalar,
                 static_cast<unsigned int>(elems));

            input0->consume(elems);
            input1->consume(elems);
            output->produce(elems);
        }

    protected:
        Fcn _fcn;
        ScalarType _scalar;
        InputPortType _inputPort0Name;
        InputPortType _inputPort1Name;
};
