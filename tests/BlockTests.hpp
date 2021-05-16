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

        auto outputs = sink.call<Pothos::BufferChunk>("feedBuffer");
        testBufferChunksEqual<OutType>(
            expectedOutputs,
            outputs);
    }
}
