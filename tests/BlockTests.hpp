// Copyright (c) 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "TestUtility.hpp"

#include <Pothos/Framework.hpp>
#include <Pothos/Testing.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

namespace VOLKTests
{
    static constexpr size_t NumRepetitions = 123;

    template <typename T>
    static constexpr T epsilon(bool)
    {
        return T(0);
    }

    template <>
    constexpr float epsilon<float>(bool lax)
    {
        return lax ? 0.5f : 1e-3f;
    }

    template <>
    constexpr double epsilon<double>(bool lax)
    {
        return lax ? 0.5 : 1e-3;
    }

    template <typename T>
    EnableIfIntegral<T, void> testBufferChunks(
        const Pothos::BufferChunk& expected,
        const Pothos::BufferChunk& actual,
        bool lax = false)
    {
        (void)lax;
        testBufferChunksEqual<T>(
            expected,
            actual);
    }

    template <typename T>
    DisableIfIntegral<T, void> testBufferChunks(
        const Pothos::BufferChunk& expected,
        const Pothos::BufferChunk& actual,
        bool lax = false)
    {
        testBufferChunksClose<T>(
            expected,
            actual,
            epsilon<T>(lax));
    }

    template <typename T>
    EnableIfComplex<T, void> testBufferChunks(
        const Pothos::BufferChunk& expected,
        const Pothos::BufferChunk& actual,
        bool lax = false)
    {
        using ScalarType = typename T::value_type;
        static const Pothos::DType dtype(typeid(ScalarType));

        auto expectedScalar = Pothos::BufferChunk(expected);
        expectedScalar.dtype = dtype;

        auto actualScalar = Pothos::BufferChunk(actual);
        actualScalar.dtype = dtype;

        testBufferChunks<ScalarType>(
            expectedScalar,
            actualScalar,
            lax);
    }

    template <typename InType, typename OutType>
    void testOneToOneBlock(
        const Pothos::Proxy& testBlock,
        const std::vector<InType>& testInputsVec,
        const std::vector<OutType>& expectedOutputsVec,
        bool lax = false,
        bool testOutputs = true)
    {
        static const Pothos::DType InDType(typeid(InType));
        static const Pothos::DType OutDType(typeid(OutType));

        const auto testInputs = VOLKTests::stdVectorToStretchedBufferChunk(
            testInputsVec,
            NumRepetitions);
        const auto expectedOutputs = VOLKTests::stdVectorToStretchedBufferChunk(
            expectedOutputsVec,
            NumRepetitions);
        if(testOutputs) POTHOS_TEST_EQUAL(testInputs.elements(), expectedOutputs.elements());

        auto source = Pothos::BlockRegistry::make("/blocks/feeder_source", InDType);
        source.call("feedBuffer", testInputs);

        auto sink = Pothos::BlockRegistry::make("/blocks/collector_sink", OutDType);

        {
            Pothos::Topology topology;
            topology.connect(source, 0, testBlock, 0);
            topology.connect(testBlock, 0, sink, 0);

            topology.commit();
            POTHOS_TEST_TRUE(topology.waitInactive(0.01));
        }

        if(testOutputs)
        {
            auto outputs = sink.call<Pothos::BufferChunk>("getBuffer");
            testBufferChunks<OutType>(
                expectedOutputs,
                outputs,
                lax);
        }
        else
        {
            auto outputs = sink.call<Pothos::BufferChunk>("getBuffer");
            POTHOS_TEST_EQUAL(OutDType, outputs.dtype);
            POTHOS_TEST_EQUAL(testInputs.elements(), outputs.elements());
        }
    }

    template <typename InType, typename OutType0, typename OutType1, typename OutputPortType>
    void testOneToTwoBlock(
        const Pothos::Proxy& testBlock,
        const std::vector<InType>& testInputsVec,
        const std::vector<OutType0>& expectedOutputs0Vec,
        const std::vector<OutType1>& expectedOutputs1Vec,
        const OutputPortType& outputPort0Name,
        const OutputPortType& outputPort1Name)
    {
        static const Pothos::DType InDType(typeid(InType));
        static const Pothos::DType OutDType0(typeid(OutType0));
        static const Pothos::DType OutDType1(typeid(OutType1));

        const auto testInputs = VOLKTests::stdVectorToStretchedBufferChunk(
            testInputsVec,
            NumRepetitions);
        const auto expectedOutputs0 = VOLKTests::stdVectorToStretchedBufferChunk(
            expectedOutputs0Vec,
            NumRepetitions);
        const auto expectedOutputs1 = VOLKTests::stdVectorToStretchedBufferChunk(
            expectedOutputs1Vec,
            NumRepetitions);
        POTHOS_TEST_EQUAL(testInputs.elements(), expectedOutputs0.elements());
        POTHOS_TEST_EQUAL(testInputs.elements(), expectedOutputs1.elements());

        auto source = Pothos::BlockRegistry::make("/blocks/feeder_source", InDType);
        source.call("feedBuffer", testInputs);

        auto sink0 = Pothos::BlockRegistry::make("/blocks/collector_sink", OutDType0);
        auto sink1 = Pothos::BlockRegistry::make("/blocks/collector_sink", OutDType1);

        {
            Pothos::Topology topology;
            topology.connect(source, 0, testBlock, 0);
            topology.connect(testBlock, outputPort0Name, sink0, 0);
            topology.connect(testBlock, outputPort1Name, sink1, 0);

            topology.commit();
            POTHOS_TEST_TRUE(topology.waitInactive(0.01));
        }

        auto outputs0 = sink0.call<Pothos::BufferChunk>("getBuffer");
        testBufferChunks<OutType0>(
            expectedOutputs0,
            outputs0);

        auto outputs1 = sink1.call<Pothos::BufferChunk>("getBuffer");
        testBufferChunks<OutType1>(
            expectedOutputs1,
            outputs1);
    }

    template <typename InType0, typename InType1, typename OutType, typename InputPortType>
    void testTwoToOneBlock(
        const Pothos::Proxy& testBlock,
        const std::vector<InType0>& testInputs0Vec,
        const std::vector<InType1>& testInputs1Vec,
        const std::vector<OutType>& expectedOutputsVec,
        const InputPortType& inputPort0Name,
        const InputPortType& inputPort1Name,
        bool lax = false)
    {
        static const Pothos::DType InDType0(typeid(InType0));
        static const Pothos::DType InDType1(typeid(InType1));
        static const Pothos::DType OutDType(typeid(OutType));

        const auto testInputs0 = VOLKTests::stdVectorToStretchedBufferChunk(
            testInputs0Vec,
            NumRepetitions);
        const auto testInputs1 = VOLKTests::stdVectorToStretchedBufferChunk(
            testInputs1Vec,
            NumRepetitions);
        const auto expectedOutputs = VOLKTests::stdVectorToStretchedBufferChunk(
            expectedOutputsVec,
            NumRepetitions);
        POTHOS_TEST_EQUAL(testInputs0.elements(), testInputs1.elements());
        POTHOS_TEST_EQUAL(testInputs0.elements(), expectedOutputs.elements());

        auto source0 = Pothos::BlockRegistry::make("/blocks/feeder_source", InDType0);
        source0.call("feedBuffer", testInputs0);

        auto source1 = Pothos::BlockRegistry::make("/blocks/feeder_source", InDType1);
        source1.call("feedBuffer", testInputs1);

        auto sink = Pothos::BlockRegistry::make("/blocks/collector_sink", OutDType);

        {
            Pothos::Topology topology;
            topology.connect(source0, 0, testBlock, inputPort0Name);
            topology.connect(source1, 0, testBlock, inputPort1Name);
            topology.connect(testBlock, 0, sink, 0);

            topology.commit();
            POTHOS_TEST_TRUE(topology.waitInactive(0.01));
        }

        auto outputs = sink.call<Pothos::BufferChunk>("getBuffer");
        testBufferChunks<OutType>(
            expectedOutputs,
            outputs,
            lax);
    }

    // Note: only use if input types are same and output types are same
    template <typename InType, typename OutType>
    void testMToNBlock(
        const Pothos::Proxy& block,
        const std::vector<std::vector<InType>>& inputs,
        const std::vector<std::vector<OutType>>& expectedOutputs,
        bool lax = false,
        bool testOutputs = true)
    {
        static const Pothos::DType inDType(typeid(InType));
        static const Pothos::DType outDType(typeid(OutType));

        std::cout << "Testing " << block.call<std::string>("getName")
                  << " (" << inDType.name() << " x" << inputs.size()
                  << " -> " << outDType.name() << " x" << expectedOutputs.size()
                  << ")..." << std::endl;

        std::vector<Pothos::BufferChunk> inputBuffers;
        std::vector<Pothos::BufferChunk> expectedOutputBuffers;

        std::transform(
            inputs.begin(),
            inputs.end(),
            std::back_inserter(inputBuffers),
            [](const std::vector<InType>& input)
            {
                return stdVectorToStretchedBufferChunk(input, NumRepetitions);
            });
        std::transform(
            expectedOutputs.begin(),
            expectedOutputs.end(),
            std::back_inserter(expectedOutputBuffers),
            [](const std::vector<InType>& expectedOutput)
            {
                return stdVectorToStretchedBufferChunk(expectedOutput, NumRepetitions);
            });

        std::vector<Pothos::Proxy> sources;
        for(size_t input = 0; input < inputs.size(); ++input)
        {
            sources.emplace_back(Pothos::BlockRegistry::make(
                "/blocks/feeder_source",
                inDType));
            sources.back().call("feedBuffer", inputBuffers[input]);
        }

        std::vector<Pothos::Proxy> sinks;
        for(size_t output = 0; output < expectedOutputs.size(); ++output)
        {
            sinks.emplace_back(Pothos::BlockRegistry::make(
                "/blocks/collector_sink",
                outDType));
        }

        {
            Pothos::Topology topology;
            for(size_t input = 0; input < sources.size(); ++input)
            {
                topology.connect(sources[input], 0, block, input);
            }
            for(size_t output = 0; output < sinks.size(); ++output)
            {
                topology.connect(block, output, sinks[output], 0);
            }

            topology.commit();
            POTHOS_TEST_TRUE(topology.waitInactive(0.01));
        }

        for(size_t output = 0; output < sinks.size(); ++output)
        {
            std::cout << " * Testing output " << output << "..." << std::endl;

            auto outputBuffer = sinks[output].call<Pothos::BufferChunk>("getBuffer");
            if(testOutputs)
            {
                testBufferChunks<OutType>(
                    expectedOutputBuffers[output],
                    outputBuffer,
                    lax);
            }
            else
            {
                POTHOS_TEST_EQUAL(outDType, outputBuffer.dtype);
                POTHOS_TEST_EQUAL(inputBuffers[0].elements(), outputBuffer.elements());
            }
        }
    }
}
