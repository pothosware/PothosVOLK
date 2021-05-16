// Copyright (c) 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "TestUtility.hpp"

#include <Pothos/Framework.hpp>
#include <Pothos/Testing.hpp>

#include <vector>

namespace VOLKTests
{
    static constexpr size_t NumRepetitions = 256;

    template <typename T>
    static constexpr T epsilon()
    {
        return T(0);
    }

    template <>
    constexpr float epsilon<float>()
    {
        return 1e-3f;
    }

    template <>
    constexpr double epsilon<double>()
    {
        return 1e-3;
    }

    template <typename T>
    DisableIfComplex<T, void> testBufferChunks(
        const Pothos::BufferChunk& expected,
        const Pothos::BufferChunk& actual)
    {
        testBufferChunksClose<T>(
            expected,
            actual,
            epsilon<T>());
    }

    template <typename T>
    EnableIfComplex<T, void> testBufferChunks(
        const Pothos::BufferChunk& expected,
        const Pothos::BufferChunk& actual)
    {
        using ScalarType = typename T::value_type;
        static const Pothos::DType dtype(typeid(ScalarType));

        auto expectedScalar = Pothos::BufferChunk(expected);
        expectedScalar.dtype = dtype;

        auto actualScalar = Pothos::BufferChunk(actual);
        actualScalar.dtype = dtype;

        testBufferChunks<ScalarType>(
            expectedScalar,
            actualScalar);
    }

    template <typename InType, typename OutType>
    void testOneToOneBlock(
        const Pothos::Proxy& testBlock,
        const std::vector<InType>& testInputsVec,
        const std::vector<OutType>& expectedOutputsVec)
    {
        static const Pothos::DType InDType(typeid(InType));
        static const Pothos::DType OutDType(typeid(OutType));

        const auto testInputs = VOLKTests::stdVectorToStretchedBufferChunk(
            testInputsVec,
            NumRepetitions);
        const auto expectedOutputs = VOLKTests::stdVectorToStretchedBufferChunk(
            expectedOutputsVec,
            NumRepetitions);

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

        auto outputs = sink.call<Pothos::BufferChunk>("getBuffer");
        testBufferChunks<OutType>(
            expectedOutputs,
            outputs);
    }

    template <typename InType0, typename InType1, typename OutType>
    void testTwoToOneBlock(
        const Pothos::Proxy& testBlock,
        const std::vector<InType0>& testInputs0Vec,
        const std::vector<InType1>& testInputs1Vec,
        const std::vector<OutType>& expectedOutputsVec)
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

        auto source0 = Pothos::BlockRegistry::make("/blocks/feeder_source", InDType0);
        source0.call("feedBuffer", testInputs0);

        auto source1 = Pothos::BlockRegistry::make("/blocks/feeder_source", InDType1);
        source1.call("feedBuffer", testInputs1);

        auto sink = Pothos::BlockRegistry::make("/blocks/collector_sink", OutDType);

        {
            Pothos::Topology topology;
            topology.connect(source0, 0, testBlock, 0);
            topology.connect(source1, 0, testBlock, 1);
            topology.connect(testBlock, 0, sink, 0);

            topology.commit();
            POTHOS_TEST_TRUE(topology.waitInactive(0.01));
        }

        auto outputs = sink.call<Pothos::BufferChunk>("getBuffer");
        testBufferChunks<OutType>(
            expectedOutputs,
            outputs);
    }
}
