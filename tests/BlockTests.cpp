// Copyright (c) 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "BlockTests.hpp"
#include "TestUtility.hpp"

#include <Pothos/Framework.hpp>
#include <Pothos/Testing.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <numeric>
#include <type_traits>

#warning TODO: block variable naming consistency, no unnecessary variables

//
// /volk/acos
//

POTHOS_TEST_BLOCK("/volk/tests", test_acos)
{
    std::vector<float> testInputs      = {0.0f, 0.5f, 1.0f};
    std::vector<float> expectedOutputs = {M_PI_2, M_PI/3.0f, 0.0f};

    auto acos = Pothos::BlockRegistry::make("/volk/acos");

    VOLKTests::testOneToOneBlock<float,float>(
        acos,
        testInputs,
        expectedOutputs);
}

//
// /volk/accumulator
//

template <typename T>
static void testAccumulator(const std::vector<T>& testValues)
{
    const Pothos::DType dtype(typeid(T));

    std::cout << " * Testing " << dtype.name() << "..." << std::endl;

    T sum = std::accumulate(
        testValues.begin(),
        testValues.end(),
        T(0));
    sum *= T(VOLKTests::NumRepetitions);

    auto accumulator = Pothos::BlockRegistry::make(
        "/volk/accumulator",
        dtype);

    VOLKTests::testOneToOneBlock<T,T>(
        accumulator,
        testValues,
        testValues);

    const auto blockSum = accumulator.call<T>("currentSum");
    POTHOS_TEST_EQUAL(sum, blockSum);
}

POTHOS_TEST_BLOCK("/volk/tests", test_accumulator)
{
    testAccumulator<float>({
        10.0f, 20.0f, 30.0f, 40.0f, 50.0f,
        60.0f, 70.0f, 80.0f, 90.0f, 100.0f
    });
    testAccumulator<std::complex<float>>({
        {10.0f,20.0f},{30.0f,40.0f},{50.0f,60.0f},
        {70.0f,80.0f},{90.0f,100.0f}
    });
}

//
// /volk/add
//

template <typename InType0, typename InType1, typename OutType>
static void getAddTestValues(
    std::vector<InType0>* pTestInputs0,
    std::vector<InType1>* pTestInputs1,
    std::vector<OutType>* pExpectedOutputs)
{
    *pTestInputs0 = {InType0(1), InType0(2), InType0(3)};
    *pTestInputs1 = {InType1(4), InType1(5), InType1(6)};
    *pExpectedOutputs = {OutType(5), OutType(7), OutType(9)};
}

template <typename InType0, typename InType1, typename OutType>
static void testAdd()
{
    Pothos::DType inDType0(typeid(InType0));
    Pothos::DType inDType1(typeid(InType1));
    Pothos::DType outDType(typeid(OutType));

    std::cout << " * Testing " << inDType0.name() << " + "
              << inDType1.name() << " = " << outDType.name()
              << "..." << std::endl;

    std::vector<InType0> testInputs0;
    std::vector<InType1> testInputs1;
    std::vector<OutType> expectedOutputs;
    getAddTestValues(
        &testInputs0,
        &testInputs1,
        &expectedOutputs);
    
    auto add = Pothos::BlockRegistry::make(
        "/volk/add",
        inDType0,
        inDType1,
        outDType);

    VOLKTests::testTwoToOneBlock<InType0,InType1,OutType,size_t>(
        add,
        testInputs0,
        testInputs1,
        expectedOutputs,
        0,
        1);
}

POTHOS_TEST_BLOCK("/volk/tests", test_add)
{
    testAdd<float,float,float>();
    testAdd<float,double,double>();
    testAdd<double,double,double>();
    testAdd<std::complex<float>,float,std::complex<float>>();
    testAdd<std::complex<float>,std::complex<float>,std::complex<float>>();
}

//
// /volk/add_quad
//

POTHOS_TEST_BLOCK("/volk/tests", test_add_quad)
{
    //
    // Test values
    //

    const std::vector<std::vector<int16_t>> inputVecs =
    {
        { 1,  2,   3,  4,  5,  6,  7,  8,  9, 10},
        {-4, -3,  -2, -1,  0,  1,  2,  3,  4,  5},
        { 4,  3,   2,  1,  0, -1, -2, -3, -4, -5},
        {-1,  2,  -3,  4, -5,  6, -7,  8, -9, 10},
        { 5,  10, 15, 20, 25, 30, 35, 40, 45, 50}
    };
    std::vector<Pothos::BufferChunk> inputs;
    std::transform(
        inputVecs.begin(),
        inputVecs.end(),
        std::back_inserter(inputs),
        [](const std::vector<int16_t>& vec)
        {
            return VOLKTests::stdVectorToStretchedBufferChunk(
                vec,
                VOLKTests::NumRepetitions);
        });

    const std::vector<std::vector<int16_t>> expectedOutputVecs =
    {
        {-3, -1,   1,  3,  5,  7,  9, 11, 13, 15},
        { 5,  5,   5,  5,  5,  5,  5,  5,  5,  5},
        { 0,  4,   0,  8,  0, 12,  0, 16,  0, 20},
        { 6,  12, 18, 24, 30, 36, 42, 48, 54, 60}
    };
    std::vector<Pothos::BufferChunk> expectedOutputs;
    std::transform(
        expectedOutputVecs.begin(),
        expectedOutputVecs.end(),
        std::back_inserter(expectedOutputs),
        [](const std::vector<int16_t>& vec)
        {
            return VOLKTests::stdVectorToStretchedBufferChunk(
                vec,
                VOLKTests::NumRepetitions);
        });

    //
    // Test implementation
    //

    auto addQuadBlock = Pothos::BlockRegistry::make("/volk/add_quad");

    std::vector<Pothos::Proxy> sources;
    std::transform(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(sources),
        [](const Pothos::BufferChunk& bufferChunk)
        {
            auto source = Pothos::BlockRegistry::make("/blocks/feeder_source", "int16");
            source.call("feedBuffer", bufferChunk);

            return source;
        });

    std::vector<Pothos::Proxy> sinks;
    for(size_t i = 0; i < expectedOutputs.size(); ++i)
    {
        sinks.emplace_back(Pothos::BlockRegistry::make("/blocks/collector_sink", "int16"));
    }

    {
        Pothos::Topology topology;
        for(size_t input = 0; input < inputs.size(); ++input)
        {
            topology.connect(
                sources[input],
                0,
                addQuadBlock,
                input);
        }
        for(size_t output = 0; output < expectedOutputs.size(); ++output)
        {
            topology.connect(
                addQuadBlock,
                output,
                sinks[output],
                0);
        }

        topology.commit();
        POTHOS_TEST_TRUE(topology.waitInactive(0.01));
    }

    for(size_t output = 0; output < expectedOutputs.size(); ++output)
    {
        std::cout << " * Testing output " << output << "..." << std::endl;

        VOLKTests::testBufferChunks<int16_t>(
            expectedOutputs[output],
            sinks[output].call<Pothos::BufferChunk>("getBuffer"));
    }
}

//
// /volk/add_scalar
//

POTHOS_TEST_BLOCK("/volk/tests", test_add_scalar)
{
    const std::vector<float> testInputs{123.4f,567.8f,901.2f,345.6f,789.0f};
    constexpr float scalar = 0.5;

    std::vector<float> expectedOutputs(testInputs);
    for(auto& val: expectedOutputs) val += scalar;

    auto addScalarBlock = Pothos::BlockRegistry::make("/volk/add_scalar");
    addScalarBlock.call("setScalar", scalar);
    POTHOS_TEST_CLOSE(scalar, addScalarBlock.call<float>("scalar"), 1e-6f);

    VOLKTests::testOneToOneBlock<float,float>(
        addScalarBlock,
        testInputs,
        expectedOutputs);
}

//
// /volk/and
//

POTHOS_TEST_BLOCK("/volk/tests", test_and)
{
    std::vector<int> testInputs0     = {123,456,789};
    std::vector<int> testInputs1     = {321,654,987};
    std::vector<int> expectedOutputs = {testInputs0[0] & testInputs1[0],
                                        testInputs0[1] & testInputs1[1],
                                        testInputs0[2] & testInputs1[2]};

    auto andBlock = Pothos::BlockRegistry::make("/volk/and");

    VOLKTests::testTwoToOneBlock<int,int,int,size_t>(
        andBlock,
        testInputs0,
        testInputs1,
        expectedOutputs,
        0,
        1);
}

//
// /volk/asin
//

POTHOS_TEST_BLOCK("/volk/tests", test_asin)
{
    std::vector<float> testInputs      = {0.0f, 0.5f, 1.0f};
    std::vector<float> expectedOutputs = {0.0f, M_PI/6.0f, M_PI_2};

    auto asin = Pothos::BlockRegistry::make("/volk/asin");

    VOLKTests::testOneToOneBlock<float,float>(
        asin,
        testInputs,
        expectedOutputs);
}

//
// /volk/atan
//

POTHOS_TEST_BLOCK("/volk/tests", test_atan)
{
    std::vector<float> testInputs      = {0.0f, 1.0f, INFINITY};
    std::vector<float> expectedOutputs = {0.0f, M_PI/4.0f, M_PI_2};

    auto atan = Pothos::BlockRegistry::make("/volk/atan");

    VOLKTests::testOneToOneBlock<float,float>(
        atan,
        testInputs,
        expectedOutputs);
}

//
// /volk/atan2
//

POTHOS_TEST_BLOCK("/volk/tests", test_atan2)
{
    constexpr float normalizationFactor = 5.0f;

    const std::vector<std::complex<float>> testInputs{
        {0.5f,1.0f},{1.5f,2.0f},{2.5f,3.0f},
        {3.5f,4.0f},{4.5f,5.0f},{5.5f,6.0f}
    };
    std::vector<float> expectedOutputs;

    std::transform(
        testInputs.begin(),
        testInputs.end(),
        std::back_inserter(expectedOutputs),
        [normalizationFactor](const std::complex<float>& input)
        {
            return std::atan2(input.imag(), input.real()) / normalizationFactor;
        });

    auto atan2 = Pothos::BlockRegistry::make("/volk/atan2");
    atan2.call("setNormalizationFactor", normalizationFactor);
    POTHOS_TEST_EQUAL(normalizationFactor, atan2.call<float>("normalizationFactor"));

    VOLKTests::testOneToOneBlock<std::complex<float>,float>(
        atan2,
        testInputs,
        expectedOutputs);
}

//
// /volk/binary_slicer
//

POTHOS_TEST_BLOCK("/volk/tests", test_binary_slicer)
{
    std::vector<float> inputs           = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
    std::vector<int8_t> expectedOutputs = {0, 0, 1, 1, 1};

    auto binarySlicerBlock = Pothos::BlockRegistry::make("/volk/binary_slicer");

    VOLKTests::testOneToOneBlock<float,int8_t>(
        binarySlicerBlock,
        inputs,
        expectedOutputs);
}

//
// /volk/byteswap
//

template <typename T>
static void testByteswap(
    const std::vector<T>& inputs,
    const std::vector<T>& expectedOutputs)
{
    const auto dtype = Pothos::DType(typeid(T));

    auto byteswapBlock = Pothos::BlockRegistry::make(
        "/volk/byteswap",
        dtype);

    VOLKTests::testOneToOneBlock<T,T>(
        byteswapBlock,
        inputs,
        expectedOutputs);
}

POTHOS_TEST_BLOCK("/volk/tests", test_byteswap)
{
    testByteswap<uint16_t>(
        {0x0102,0x0304,0x0506},
        {0x0201,0x0403,0x0605});
    testByteswap<uint32_t>(
        {0x01020304,0x03040506,0x05060708},
        {0x04030201,0x06050403,0x08070605});
    testByteswap<uint64_t>(
        {0x0102030405060708,0x030405060708090A,0x05060708090A0B0C},
        {0x0807060504030201,0x0A09080706050403,0x0C0B0A0908070605});
}

//
// /volk/calc_spectral_noise_floor
//

POTHOS_TEST_BLOCK("/volk/tests", test_calc_spectral_noise_floor)
{
    constexpr float spectralExclusionValue = 5.0f;

    auto calcSpectralNoiseFloorBlock = Pothos::BlockRegistry::make("/volk/calc_spectral_noise_floor");
    calcSpectralNoiseFloorBlock.call(
        "setSpectralExclusionValue",
        spectralExclusionValue);
    POTHOS_TEST_EQUAL(
        spectralExclusionValue,
        calcSpectralNoiseFloorBlock.call<float>("spectralExclusionValue"));

    // Just make sure the block executes
    VOLKTests::testOneToOneBlock<float,float>(
        calcSpectralNoiseFloorBlock,
        {0,1,2,3,4,5,6,7,8,9,10},
        {},
        false /*lax*/,
        false /*testOutputs*/);
}

//
// /volk/conjugate
//

POTHOS_TEST_BLOCK("/volk/tests", test_conjugate)
{
    auto conjugateBlock = Pothos::BlockRegistry::make("/volk/conjugate");

    VOLKTests::testOneToOneBlock<std::complex<float>,std::complex<float>>(
        conjugateBlock,
        {{0.0f,1.0f},{2.0f,3.0f},{4.0f,5.0f}},
        {{0.0f,-1.0f},{2.0f,-3.0f},{4.0f,-5.0f}});
}

//
// /volk/convert
//

template <typename InType, typename OutType>
static void testConvert(
    const std::vector<InType>& inputs,
    const std::vector<OutType>& expectedOutputs)
{
    const Pothos::DType inDType(typeid(InType));
    const Pothos::DType outDType(typeid(OutType));

    std::cout << " * Testing " << inDType.name()
              << " -> " << outDType.name()
              << "..." << std::endl;

    auto convertBlock = Pothos::BlockRegistry::make(
        "/volk/convert",
        inDType,
        outDType);

    VOLKTests::testOneToOneBlock<InType,OutType>(
        convertBlock,
        inputs,
        expectedOutputs);
}

POTHOS_TEST_BLOCK("/volk/tests", test_convert)
{
    testConvert<int8_t,int16_t>(
        {0, 1, 2, 3, 4, 5, 127},
        {0, 256, 512, 768, 1024, 1280, 32512});

    testConvert<int16_t,int8_t>(
        {0, 256, 512, 768, 1024, 1280, 32512},
        {0, 1, 2, 3, 4, 5, 127});
}

//
// /volk/convert_scaled
//

template <typename InType, typename OutType>
static void testConvertScaled(
    const std::vector<InType>& inputs,
    const std::vector<OutType>& expectedOutputs,
    float scalar)
{
    const Pothos::DType inDType(typeid(InType));
    const Pothos::DType outDType(typeid(OutType));

    std::cout << " * Testing " << inDType.name()
              << " -> " << outDType.name()
              << "..." << std::endl;

    auto convertScaledBlock = Pothos::BlockRegistry::make(
        "/volk/convert_scaled",
        inDType,
        outDType);
    convertScaledBlock.call("setScalar", scalar);
    POTHOS_TEST_CLOSE(
        scalar,
        convertScaledBlock.call<float>("scalar"),
        1e-6f);

    VOLKTests::testOneToOneBlock<InType,OutType>(
        convertScaledBlock,
        inputs,
        expectedOutputs);
}

POTHOS_TEST_BLOCK("/volk/tests", test_convert_scaled)
{
    testConvertScaled<float,int8_t>(
        {0.01f, 0.25f, 0.03f, 0.45f, 0.05f},
        {1,     25,    3,     45,    5},
        100.0f);
    testConvertScaled<float,int16_t>(
        {0.1f, 0.25f, 0.3f, 0.045f, 3.0f},
        {1000, 2500,  3000, 450,    30000},
        10000.0f);
    testConvertScaled<float,int32_t>(
        {1.5e1, 2.5e2, 3.5e3,  4.25e4,  5e5},
        {1500,  25000, 350000, 4250000, 50000000},
        100.0f);
    testConvertScaled<int8_t,float>(
        {1,     25,    3,     45,    5},
        {0.01f, 0.25f, 0.03f, 0.45f, 0.05f},
        100.0f);
    testConvertScaled<int16_t,float>(
        {1000, 2500,  3000, 450,    30000},
        {0.1f, 0.25f, 0.3f, 0.045f, 3.0f},
        10000.0f);
    testConvertScaled<int32_t,float>(
        {1500,  25000, 350000, 4250000, 50000000},
        {1.5e1, 2.5e2, 3.5e3,  4.25e4,  5e5},
        100.0f);
}

//
// /volk/cos
//

POTHOS_TEST_BLOCK("/volk/tests", test_cos)
{
    auto cosBlock = Pothos::BlockRegistry::make("/volk/cos");

    VOLKTests::testOneToOneBlock<float,float>(
        cosBlock,
        {0.0f, M_PI_2, M_PI},
        {1.0f, 0.0f, -1.0f});
}

//
// /volk/deinterleave
//

template <typename InType, typename OutType>
static void testDeinterleave(
    const std::vector<InType>& inputs,
    const std::vector<OutType>& expectedOutputs0,
    const std::vector<OutType>& expectedOutputs1)
{
    const Pothos::DType inDType(typeid(InType));
    const Pothos::DType outDType(typeid(OutType));

    std::cout << " * Testing " << inDType.name()
              << " -> " << outDType.name()
              << "..." << std::endl;

    auto deinterleaveBlock = Pothos::BlockRegistry::make(
        "/volk/deinterleave",
        inDType,
        outDType);

    VOLKTests::testOneToTwoBlock<InType,OutType,OutType,std::string>(
        deinterleaveBlock,
        inputs,
        expectedOutputs0,
        expectedOutputs1,
        "real",
        "imag");
}

POTHOS_TEST_BLOCK("/volk/tests", test_deinterleave)
{
    testDeinterleave<std::complex<int8_t>,int16_t>(
        {{-4,-3}, {-2,-1}, {0,1}, {2,3}, {4,5}},
        {-1024,   -512,    0,     512,   1024},
        {-768,    -256,    256,   768,   1280});

    testDeinterleave<std::complex<int16_t>,int16_t>(
        {{-10000,-1000}, {-100,-10}, {10,100}, {1000,10000}},
        {-10000,         -100,       10,       1000},
        {-1000,          -10,        100,      10000});

    testDeinterleave<std::complex<float>,float>(
        {{-2.5f,-1.5f}, {-0.5f,0.5f}, {1.5f,2.5f}},
        {-2.5f,         -0.5f,        1.5f},
        {-1.5f,         0.5f,         2.5f});

    testDeinterleave<std::complex<float>,double>(
        {{-2.5f,-1.5f}, {-0.5f,0.5f}, {1.5f,2.5f}},
        {-2.5,          -0.5,         1.5},
        {-1.5,          0.5,          2.5});
}

//
// /volk/deinterleave_imag
//

POTHOS_TEST_BLOCK("/volk/tests", test_deinterleave_imag)
{
    auto deinterleaveImagBlock = Pothos::BlockRegistry::make("/volk/deinterleave_imag");

    VOLKTests::testOneToOneBlock<std::complex<float>,float>(
        deinterleaveImagBlock,
        {{-2.5f,-1.5f}, {-0.5f,0.5f}, {1.5f,2.5f}},
        {-1.5f,         0.5f,         2.5f});
}

//
// /volk/deinterleave_real
//

template <typename InType, typename OutType>
static void testDeinterleaveReal(
    const std::vector<InType>& inputs,
    const std::vector<OutType>& expectedOutputs)
{
    const Pothos::DType inDType(typeid(InType));
    const Pothos::DType outDType(typeid(OutType));

    std::cout << " * Testing " << inDType.name()
              << " -> " << outDType.name()
              << "..." << std::endl;

    auto deinterleaveBlock = Pothos::BlockRegistry::make(
        "/volk/deinterleave_real",
        inDType,
        outDType);

    VOLKTests::testOneToOneBlock<InType,OutType>(
        deinterleaveBlock,
        inputs,
        expectedOutputs);
}

POTHOS_TEST_BLOCK("/volk/tests", test_deinterleave_real)
{
    testDeinterleaveReal<std::complex<int8_t>,int8_t>(
        {{-4,-3}, {-2,-1}, {0,1}, {2,3}, {4,5}},
        {-4,      -2,      0,     2,     4});

    testDeinterleaveReal<std::complex<int8_t>,int16_t>(
        {{-4,-3}, {-2,-1}, {0,1}, {2,3}, {4,5}},
        {-512,    -256,    0,     256,   512});

    testDeinterleaveReal<std::complex<int16_t>,int8_t>(
        {{16384,-8192}, {-4096,2048}, {1024,-512}, {-256,128}},
        {64,            -16,          4,           -1});

    testDeinterleaveReal<std::complex<int16_t>,int16_t>(
        {{-10000,-1000}, {-100,-10}, {10,100}, {1000,10000}},
        {-10000,         -100,       10,       1000});

    testDeinterleaveReal<std::complex<float>,float>(
        {{-2.5f,-1.5f}, {-0.5f,0.5f}, {1.5f,2.5f}},
        {-2.5f,         -0.5f,        1.5f});

    testDeinterleaveReal<std::complex<float>,double>(
        {{-2.5f,-1.5f}, {-0.5f,0.5f}, {1.5f,2.5f}},
        {-2.5,          -0.5,         1.5});
}

//
// /volk/deinterleave_real_scaled
//

template <typename InType, typename OutType>
static void testDeinterleaveRealScaled(
    const std::vector<InType>& inputs,
    const std::vector<OutType>& expectedOutputs,
    float scalar)
{
    const Pothos::DType inDType(typeid(InType));
    const Pothos::DType outDType(typeid(OutType));

    std::cout << " * Testing " << inDType.name()
              << " -> " << outDType.name()
              << "..." << std::endl;

    auto deinterleaveRealScaledBlock = Pothos::BlockRegistry::make(
        "/volk/deinterleave_real_scaled",
        inDType,
        outDType);
    deinterleaveRealScaledBlock.call("setScalar", scalar);
    POTHOS_TEST_CLOSE(
        scalar,
        deinterleaveRealScaledBlock.call<float>("scalar"),
        1e-6f);

    VOLKTests::testOneToOneBlock<InType,OutType>(
        deinterleaveRealScaledBlock,
        inputs,
        expectedOutputs);
}

POTHOS_TEST_BLOCK("/volk/tests", test_deinterleave_real_scaled)
{
    testDeinterleaveRealScaled<std::complex<float>,int16_t>(
        {{0.123f,0.0f}, {0.456f,0.0f}, {0.789f,0.0f}, {1.0f,0.0f}, {0.0f,0.0f}},
        {123,           456,           789,           1000,        0},
        1000.0f);

    testDeinterleaveRealScaled<std::complex<int8_t>,float>(
        {{10,0}, {20,0}, {30,0}, {40,0}, {50,0}},
        {0.1f,   0.2f,   0.3f,   0.4f,   0.5f},
        100.0f);

    testDeinterleaveRealScaled<std::complex<int16_t>,float>(
        {{10,0}, {20,0}, {30,0}, {40,0}, {50,0}},
        {0.1f,   0.2f,   0.3f,   0.4f,   0.5f},
        100.0f);
}

//
// /volk/deinterleave_scaled
//

template <typename InType, typename OutType>
static void testDeinterleaveScaled(
    const std::vector<InType>& inputs,
    const std::vector<OutType>& expectedOutputs0,
    const std::vector<OutType>& expectedOutputs1,
    float scalar)
{
    const Pothos::DType inDType(typeid(InType));
    const Pothos::DType outDType(typeid(OutType));

    std::cout << " * Testing " << inDType.name()
              << " -> " << outDType.name()
              << "..." << std::endl;

    auto deinterleaveScaledBlock = Pothos::BlockRegistry::make(
        "/volk/deinterleave_scaled",
        inDType,
        outDType);
    deinterleaveScaledBlock.call("setScalar", scalar);
    POTHOS_TEST_CLOSE(
        scalar,
        deinterleaveScaledBlock.call<float>("scalar"),
        1e-6f);

    VOLKTests::testOneToTwoBlock<InType,OutType,OutType,std::string>(
        deinterleaveScaledBlock,
        inputs,
        expectedOutputs0,
        expectedOutputs1,
        "real",
        "imag");
}

POTHOS_TEST_BLOCK("/volk/tests", test_deinterleave_scaled)
{
    testDeinterleaveScaled<std::complex<int8_t>,float>(
        {{-4,-3}, {-2,-1}, {0,1}, {2,3}, {4,5}},
        {-0.04f,  -0.02f,  0.0f,  0.02f, 0.04f},
        {-0.03f,  -0.01f,  0.01f, 0.03f, 0.05f},
        100.0f);

    testDeinterleaveScaled<std::complex<int16_t>,float>(
        {{-4,-3}, {-2,-1}, {0,1}, {2,3}, {4,5}},
        {-0.04f,  -0.02f,  0.0f,  0.02f, 0.04f},
        {-0.03f,  -0.01f,  0.01f, 0.03f, 0.05f},
        100.0f);
}

//
// /volk/divide
//

template <typename T>
static void testDivide(
    const std::vector<T>& inputs0,
    const std::vector<T>& inputs1,
    const std::vector<T>& expectedOutputs)
{
    const Pothos::DType dtype(typeid(T));

    std::cout << "Testing " << dtype.name() << "..." << std::endl;

    auto divideBlock = Pothos::BlockRegistry::make(
        "/volk/divide",
        dtype);

    VOLKTests::testTwoToOneBlock<T,T,T,size_t>(
        divideBlock,
        inputs0,
        inputs1,
        expectedOutputs,
        0,
        1);
}

POTHOS_TEST_BLOCK("/volk/tests", test_divide)
{
    testDivide<float>(
        {-3.0f, -2.0f,  -1.0f,  1.0f,    2.0f, 3.0f},
        {0.5f,  -0.25f, 0.125f, -8.0f,   4.0f, -2.0f},
        {-6.0f, 8.0f,   -8.0f,  -0.125f, 0.5f, -1.5f});

    testDivide<std::complex<float>>(
        {{-3.0f,-2.0f},  {-1.0f,1.0f},          {2.0f,3.0f}},
        {{0.5f,-0.25f},  {0.125f,-8.0f},        {4.0f,-2.0f}},
        {{-3.2f,-5.6f},  {-0.12692f,-0.12301f}, {0.1f,0.8f}});
}

//
// /volk/exp
//

static void testExp(const std::string& mode, bool lax)
{
    std::cout << "Testing " << mode << " mode..." << std::endl;

    VOLKTests::testOneToOneBlock<float,float>(
        Pothos::BlockRegistry::make("/volk/exp", mode),
        {0.0f, 1.0f},
        {1.0f, M_E},
        lax);
}

POTHOS_TEST_BLOCK("/volk/tests", test_exp)
{
    testExp("PRECISE", false);
    testExp("FAST", true);
}

//
// /volk/interleave
//

POTHOS_TEST_BLOCK("/volk/tests", test_interleave)
{
    auto interleaveBlock = Pothos::BlockRegistry::make("/volk/interleave");

    VOLKTests::testTwoToOneBlock<float,float,std::complex<float>,std::string>(
        interleaveBlock,
        {-2.5f,         -0.5f,        1.5f},
        {-1.5f,         0.5f,         2.5f},
        {{-2.5f,-1.5f}, {-0.5f,0.5f}, {1.5f,2.5f}},
        "real",
        "imag");
}

//
// /volk/interleave_scaled
//

POTHOS_TEST_BLOCK("/volk/tests", test_interleave_scaled)
{
    constexpr float scalar = 100.0f;

    auto interleaveScaledBlock = Pothos::BlockRegistry::make("/volk/interleave_scaled");
    interleaveScaledBlock.call("setScalar", scalar);
    POTHOS_TEST_CLOSE(
        scalar,
        interleaveScaledBlock.call<float>("scalar"),
        1e-6f);

    VOLKTests::testTwoToOneBlock<float,float,std::complex<int16_t>,std::string>(
        interleaveScaledBlock,
        {-2.5f,       -0.5f,    1.5f},
        {-1.5f,       0.5f,     2.5f},
        {{-250,-150}, {-50,50}, {150,250}},
        "real",
        "imag");
}

//
// /volk/invsqrt
//

POTHOS_TEST_BLOCK("/volk/tests", test_invsqrt)
{
    auto invSqrtBlock = Pothos::BlockRegistry::make("/volk/invsqrt");

    VOLKTests::testOneToOneBlock<float,float>(
        invSqrtBlock,
        {0.125f,    0.5f,      2.0f},
        {2.828427f, 1.414213f, 0.707106f});
}

//
// /volk/log2
//

POTHOS_TEST_BLOCK("/volk/tests", test_log2)
{
    auto log2Block = Pothos::BlockRegistry::make("/volk/log2");

    VOLKTests::testOneToOneBlock<float,float>(
        log2Block,
        {1.0f, 2.0f, 4.0f, 5.0f},
        {0.0f, 1.0f, 2.0f, 2.321928f});
}

//
// /volk/magnitude
//

template <typename T>
static void testMagnitude(
    const std::vector<std::complex<T>>& inputs,
    const std::vector<T>& expectedOutputs)
{
    const Pothos::DType dtype(typeid(T));

    std::cout << "Testing " << dtype.name() << "..." << std::endl;

    VOLKTests::testOneToOneBlock<std::complex<T>,T>(
        Pothos::BlockRegistry::make("/volk/magnitude", dtype),
        inputs,
        expectedOutputs);
}

POTHOS_TEST_BLOCK("/volk/tests", test_magnitude)
{
    const std::vector<std::complex<int16_t>> int16Inputs =
    {
        {0,5}, {10,15}, {20,25}, {30,35}, {40,45}
    };
    std::vector<int16_t> int16ExpectedOutputs;
    std::transform(
        int16Inputs.begin(),
        int16Inputs.end(),
        std::back_inserter(int16ExpectedOutputs),
        [](const std::complex<int16_t>& input)
        {
            static constexpr float scalar = std::numeric_limits<int16_t>::max();

            std::complex<float> scaledInput{
                float(input.real()) / scalar,
                float(input.imag()) / scalar};

            return int16_t(std::abs(scaledInput) * scalar);
        });

    const std::vector<std::complex<float>> floatInputs =
    {
        {1.23f,4.56f}, {78.9f,12.3f}, {456.0f,789.0f}
    };
    std::vector<float> floatExpectedOutputs;
    std::transform(
        floatInputs.begin(),
        floatInputs.end(),
        std::back_inserter(floatExpectedOutputs),
        [](const std::complex<float>& input){return std::abs(input);});

    testMagnitude<int16_t>(
       int16Inputs,
       int16ExpectedOutputs);

    testMagnitude<float>(
       floatInputs,
       floatExpectedOutputs);
}

//
// /volk/magnitude_squared
//

POTHOS_TEST_BLOCK("/volk/tests", test_magnitude_squared)
{
    const std::vector<std::complex<float>> inputs =
    {
        {1.23f,4.56f}, {78.9f,12.3f}, {456.0f,789.0f}
    };
    std::vector<float> expectedOutputs;
    std::transform(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(expectedOutputs),
        [](const std::complex<float>& input){return std::pow(std::abs(input), 2.0f);});

    VOLKTests::testOneToOneBlock<std::complex<float>,float>(
        Pothos::BlockRegistry::make("/volk/magnitude_squared"),
        inputs,
        expectedOutputs);
}

//
// /volk/max
//

template <typename T>
static void testMax()
{
    const Pothos::DType dtype(typeid(T));

    std::cout << "Testing " << dtype.name() << "..." << std::endl;

    auto maxBlock = Pothos::BlockRegistry::make("/volk/max", dtype);

    VOLKTests::testTwoToOneBlock<T,T,T,size_t>(
        maxBlock,
        {-5.0f, 3.0f,  -1.0f, 1.0f, -3.0f, 5.0f},
        {4.0f,  -2.0f, 0.0f,  2.0f, -4.0f, 6.0f},
        {4.0f,  3.0f,  0.0f,  2.0f, -3.0f, 6.0f},
        0,
        1);
}

POTHOS_TEST_BLOCK("/volk/tests", test_max)
{
    testMax<float>();
    testMax<double>();
}

//
// /volk/max_star
//

POTHOS_TEST_BLOCK("/volk/tests", test_max_star)
{
    VOLKTests::testOneToOneBlock<int16_t,int16_t>(
        Pothos::BlockRegistry::make("/volk/max_star"),
        {1,2,3,4,5},
        {},
        false /*lax*/,
        false /*testOutputs*/);
}

//
// /volk/min
//

template <typename T>
static void testMin()
{
    const Pothos::DType dtype(typeid(T));

    std::cout << "Testing " << dtype.name() << "..." << std::endl;

    auto minBlock = Pothos::BlockRegistry::make("/volk/min", dtype);

    VOLKTests::testTwoToOneBlock<T,T,T,size_t>(
        minBlock,
        {-5.0f, 3.0f,  -1.0f, 1.0f, -3.0f, 5.0f},
        {4.0f,  -2.0f, 0.0f,  2.0f, -4.0f, 6.0f},
        {-5.0f, -2.0f, -1.0f, 1.0f, -4.0f, 5.0f},
        0,
        1);
}

POTHOS_TEST_BLOCK("/volk/tests", test_min)
{
    testMin<float>();
    testMin<double>();
}

//
// /volk/mod_range
//

POTHOS_TEST_BLOCK("/volk/tests", test_mod_range)
{
    constexpr float lowerBound = 1.0f;
    constexpr float upperBound = 2.0f;
    constexpr float offset = 0.1f;

    auto modRangeBlock = Pothos::BlockRegistry::make("/volk/mod_range");

    modRangeBlock.call("setLowerBound", lowerBound);
    POTHOS_TEST_CLOSE(
        lowerBound,
        modRangeBlock.call<float>("lowerBound"),
        1e-6f);

    modRangeBlock.call("setUpperBound", upperBound);
    POTHOS_TEST_CLOSE(
        upperBound,
        modRangeBlock.call<float>("upperBound"),
        1e-6f);

    const std::vector<float> inputs =
    {
        lowerBound,
        upperBound,
        lowerBound - offset,
        lowerBound + offset,
        upperBound - offset,
        upperBound + offset
    };
    const std::vector<float> expectedOutputs =
    {
        lowerBound,
        upperBound,
        upperBound - offset,
        lowerBound + offset,
        upperBound - offset,
        lowerBound + offset
    };

    VOLKTests::testOneToOneBlock<float,float>(
        modRangeBlock,
        inputs,
        expectedOutputs);
}

//
// /volk/multiply
//

template <typename InType0, typename InType1, typename OutType>
static void testMultiply(
    const std::vector<InType0>& inputs0,
    const std::vector<InType1>& inputs1,
    const std::vector<OutType>& expectedOutputs)
{
    const Pothos::DType inDType0(typeid(InType0));
    const Pothos::DType inDType1(typeid(InType1));
    const Pothos::DType outDType(typeid(OutType));

    std::cout << "Testing " << inDType0.name() << " * "
                            << inDType1.name() << " -> "
                            << outDType.name() << "..." << std::endl;

    auto multiplyBlock = Pothos::BlockRegistry::make(
        "/volk/multiply",
        inDType0,
        inDType1,
        outDType);

    VOLKTests::testTwoToOneBlock<InType0,InType1,OutType,size_t>(
        multiplyBlock,
        inputs0,
        inputs1,
        expectedOutputs,
        0,
        1);
}

POTHOS_TEST_BLOCK("/volk/tests", test_multiply)
{
    testMultiply<float,double,double>(
        {0.5f, 1.0f, 1.5f, 2.0f, 2.5f},
        {1.0,  1.5,  2.0,  2.5,  3.5},
        {0.5,  1.5,  3.0,  5.0,  8.75});

    testMultiply<double,double,double>(
        {0.5, 1.0, 1.5, 2.0, 2.5},
        {1.0, 1.5, 2.0, 2.5, 3.5},
        {0.5, 1.5, 3.0, 5.0, 8.75});

    testMultiply<std::complex<int16_t>,std::complex<int16_t>,std::complex<int16_t>>(
        {{0,1},   {2,3},   {4,5},   {6,7},    {8,9}},
        {{-9,-8}, {-7,-6}, {-5,-4}, {-3,-2},  {-1,0}},
        {{8,-9},  {4,-33}, {0,-41}, {-4,-33}, {-8,-9}});

    testMultiply<std::complex<float>,std::complex<float>,std::complex<float>>(
        {{-2.5f,-2.0f},   {-1.5f,-1.0f},  {-0.5f,0.5f},     {1.0f,1.5f},    {2.0f,2.5f}},
        {{5.0f,1.0f},     {3.0f,0.5f},    {1.0f,-0.25f},    {-0.5f,-0.75f}, {-5.0f,-1.25f}},
        {{-10.5f,-12.5f}, {-4.0f,-3.75f}, {-0.375f,0.625f}, {0.625f,-1.5f}, {-6.875f,-15.0f}});

    testMultiply<std::complex<float>,float,std::complex<float>>(
        {{-2.5f,-2.0f}, {-1.5f,-1.0f},  {-0.5f,0.5f}, {1.0f,1.5f},  {2.0f,2.5f}},
        {1.0f,          1.5f,           2.0f,         2.5f,         3.5f},
        {{-2.5f,-2.0f}, {-2.25f,-1.5f}, {-1.0f,1.0f}, {2.5f,3.75f}, {7.0f,8.75f}});
}

//
// /volk/multiply_conjugate
//

template <typename InType, typename OutType>
static void testMultiplyConjugate(
    const std::vector<InType>& inputs0,
    const std::vector<InType>& inputs1,
    const std::vector<OutType>& expectedOutputs)
{
    const Pothos::DType inDType(typeid(InType));
    const Pothos::DType outDType(typeid(OutType));

    std::cout << "Testing " << inDType.name() << " -> "
                            << outDType.name() << "..." << std::endl;

    auto multiplyConjugateBlock = Pothos::BlockRegistry::make(
        "/volk/multiply_conjugate",
        inDType,
        outDType);

    VOLKTests::testTwoToOneBlock<InType,InType,OutType,size_t>(
        multiplyConjugateBlock,
        inputs0,
        inputs1,
        expectedOutputs,
        0,
        1);
}

POTHOS_TEST_BLOCK("/volk/tests", test_multiply_conjugate)
{
    testMultiplyConjugate<std::complex<int8_t>,std::complex<int16_t>>(
        {{0,1},   {2,3},    {4,5},    {6,7},    {8,9}},
        {{-9,-8}, {-7,-6},  {-5,-4},  {-3,-2},  {-1,0}},
        {{-8,-9}, {-32,-9}, {-40,-9}, {-32,-9}, {-8,-9}});

    testMultiplyConjugate<std::complex<float>,std::complex<float>>(
        {{-2.5f,-2.0f},  {-1.5f,-1.0f},  {-0.5f,0.5f},     {1.0f,1.5f},    {2.0f,2.5f}},
        {{5.0f,1.0f},    {3.0f,0.5f},    {1.0f,-0.25f},    {-0.5f,-0.75f}, {-5.0f,-1.25f}},
        {{-14.5f,-7.5f}, {-5.0f,-2.25f}, {-0.625f,0.375f}, {-1.625f,0.0f}, {-13.125f,-10.0f}});
}

//
// /volk/multiply_conjugate_add
//

POTHOS_TEST_BLOCK("/volk/tests", test_multiply_conjugate_add)
{
    constexpr std::complex<float> scalar{2.0f,0.5f};

    auto multiplyConjugateAddBlock = Pothos::BlockRegistry::make("/volk/multiply_conjugate_add");

    multiplyConjugateAddBlock.call("setScalar", scalar);
    POTHOS_TEST_EQUAL(
        multiplyConjugateAddBlock.call<std::complex<float>>("scalar"),
        scalar);

    const std::vector<std::complex<float>> inputs0 =
    {
        {-2.5f,-2.0f}, {-1.5f,-1.0f}, {-0.5f,0.5f}, {1.0f,1.5f}, {2.0f,2.5f}
    };
    const std::vector<std::complex<float>> inputs1 =
    {
        {5.0f,1.0f}, {3.0f,0.5f}, {1.0f,-0.25f}, {-0.5f,-0.75f}, {-5.0f,-1.25f}
    };
    std::vector<std::complex<float>> expectedOutputs;
    for(size_t i = 0; i < inputs0.size(); ++i)
    {
        expectedOutputs.emplace_back(inputs0[i] + (std::conj(inputs1[i]) * scalar));
    }

    VOLKTests::testTwoToOneBlock<std::complex<float>,std::complex<float>,std::complex<float>,size_t>(
        multiplyConjugateAddBlock,
        inputs0,
        inputs1,
        expectedOutputs,
        0,
        1);
}

//
// /volk/multiply_conjugate_scaled
//

POTHOS_TEST_BLOCK("/volk/tests", test_multiply_conjugate_scaled)
{
    constexpr float scalar = 10.0f;

    auto multiplyConjugateScaledBlock = Pothos::BlockRegistry::make("/volk/multiply_conjugate_scaled");

    multiplyConjugateScaledBlock.call("setScalar", scalar);
    POTHOS_TEST_EQUAL(
        multiplyConjugateScaledBlock.call<std::complex<float>>("scalar"),
        scalar);

    const std::vector<std::complex<int8_t>> inputs0 =
    {
        {{0,1},   {2,3},    {4,5},    {6,7},    {8,9}},
    };
    const std::vector<std::complex<int8_t>> inputs1 =
    {
        {{-9,-8}, {-7,-6},  {-5,-4},  {-3,-2},  {-1,0}},
    };
    std::vector<std::complex<float>> expectedOutputs;
    for(size_t i = 0; i < inputs0.size(); ++i)
    {
        const std::complex<float> inputs0Float(inputs0[i].real(), inputs0[i].imag());
        const std::complex<float> inputs1Float(inputs1[i].real(), inputs1[i].imag());

        expectedOutputs.emplace_back(inputs0Float * (std::conj(inputs1Float) / scalar));
    }

    VOLKTests::testTwoToOneBlock<std::complex<int8_t>,std::complex<int8_t>,std::complex<float>,size_t>(
        multiplyConjugateScaledBlock,
        inputs0,
        inputs1,
        expectedOutputs,
        0,
        1);
}

//
// /volk/multiply_scalar
//

template <typename T>
static void testMultiplyScalar(
    const std::vector<T>& inputs,
    const T& scalar)
{
    const Pothos::DType dtype(typeid(T));

    std::cout << "Testing " << dtype.name() << "..." << std::endl;

    std::vector<T> expectedOutputs;
    std::transform(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(expectedOutputs),
        [&scalar](const T& input){ return (input * scalar); });

    auto multiplyScalarBlock = Pothos::BlockRegistry::make(
        "/volk/multiply_scalar",
        dtype);

    multiplyScalarBlock.call("setScalar", scalar);
    POTHOS_TEST_EQUAL(scalar, multiplyScalarBlock.call<T>("scalar"));

    VOLKTests::testOneToOneBlock<T,T>(
        multiplyScalarBlock,
        inputs,
        expectedOutputs);
}

POTHOS_TEST_BLOCK("/volk/tests", test_multiply_scalar)
{
    testMultiplyScalar<float>(
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f},
        0.123f);

    testMultiplyScalar<std::complex<float>>(
        {{0.1f,0.2f}, {0.3f,0.4f}, {0.5f,0.6f}, {0.7f,0.8f}, {0.9f,1.0f}},
        {0.123f,0.456f});
}

//
// /volk/normalize
//

POTHOS_TEST_BLOCK("/volk/tests", test_normalize)
{
    const float scalar = 10.0f;

    auto normalizeBlock = Pothos::BlockRegistry::make("/volk/normalize");

    normalizeBlock.call("setScalar", scalar);
    POTHOS_TEST_CLOSE(
        scalar,
        normalizeBlock.call<float>("scalar"),
        1e-6f);

    VOLKTests::testOneToOneBlock<float,float>(
        normalizeBlock,
        {0.0f, 0.75f,  1.25f,  2.0f, 2.75f,  3.5f},
        {0.0f, 0.075f, 0.125f, 0.2f, 0.275f, 0.35f});
}

//
// /volk/or
//

POTHOS_TEST_BLOCK("/volk/tests", test_or)
{
    std::vector<int> testInputs0     = {123,456,789};
    std::vector<int> testInputs1     = {321,654,987};
    std::vector<int> expectedOutputs = {testInputs0[0] | testInputs1[0],
                                        testInputs0[1] | testInputs1[1],
                                        testInputs0[2] | testInputs1[2]};

    auto orBlock = Pothos::BlockRegistry::make("/volk/or");

    VOLKTests::testTwoToOneBlock<int,int,int,size_t>(
        orBlock,
        testInputs0,
        testInputs1,
        expectedOutputs,
        0,
        1);
}

//
// /volk/popcnt
//

POTHOS_TEST_BLOCK("/volk/tests", test_popcnt)
{
    VOLKTests::testOneToOneBlock<uint64_t,uint64_t>(
        Pothos::BlockRegistry::make("/volk/popcnt"),
        {0,0b101010101010101,std::numeric_limits<uint64_t>::max()},
        {0,8,64});
}

//
// /volk/pow
//

POTHOS_TEST_BLOCK("/volk/tests", test_pow)
{
    VOLKTests::testTwoToOneBlock<float,float,float,std::string>(
        Pothos::BlockRegistry::make("/volk/pow"),
        {0.5f, 1.0f, 1.5f,     2.0f,  2.5f},
        {1.0f, 1.5f, 2.0f,     2.5f,  3.0f},
        {1.0f, 1.5f, 2.82843f, 6.25f, 15.58846f},
        "exp",
        "input",
        true);
}

//
// /volk/power
//

POTHOS_TEST_BLOCK("/volk/tests", test_power)
{
    constexpr float power = 2.0f;

    auto powerBlock = Pothos::BlockRegistry::make("/volk/power");
    powerBlock.call("setPower", power);
    POTHOS_TEST_CLOSE(power, powerBlock.call<float>("power"), 1e-6f);

    VOLKTests::testOneToOneBlock<float,float>(
        powerBlock,
        {0.0f, 0.5f,  1.0f, 1.5f,  2.0f, 2.5f,  3.0f, 3.5f,   4.0f},
        {0.0f, 0.25f, 1.0f, 2.25f, 4.0f, 6.25f, 9.0f, 12.25f, 16.0f});
}

// TODO: /volk/power_spectral_density
// TODO: /volk/power_spectrum
// TODO: /volk/quad_max_star

//
// /volk/reverse
//

POTHOS_TEST_BLOCK("/volk/tests", test_reverse)
{
    static const auto reverse = [](uint32_t num) -> uint32_t
    {
        uint32_t reverse_num = 0;

        for(uint32_t i = 0; i < 32; ++i)
        {
            uint32_t temp = (num & (1 << i));
            if(temp)
                reverse_num |= (1 << (31 - i));
        }

        return reverse_num;
    };

    VOLKTests::testOneToOneBlock<uint32_t,uint32_t>(
        Pothos::BlockRegistry::make("/volk/reverse"),
        {1, 2, 3, 4, 5},
        {reverse(1), reverse(2), reverse(3), reverse(4), reverse(5)});
}

//
// /volk/sin
//

POTHOS_TEST_BLOCK("/volk/tests", test_sin)
{
    VOLKTests::testOneToOneBlock<float,float>(
        Pothos::BlockRegistry::make("/volk/sin"),
        {0.0f, M_PI_2, M_PI},
        {0.0f, 1.0f,   0.0f});
}

//
// /volk/square_dist
//

POTHOS_TEST_BLOCK("/volk/tests", test_square_dist)
{
    auto squareDist = Pothos::BlockRegistry::make("/volk/square_dist");
    POTHOS_TEST_CLOSE(1.0f, squareDist.call<float>("scalar"), 1e-6f);

    const std::complex<float> complexInput{0.5f, 2.0f};
    squareDist.call("setComplexInput", complexInput);
    POTHOS_TEST_EQUAL(
        complexInput,
        squareDist.call<std::complex<float>>("complexInput"));

    constexpr size_t N = 16;
    std::vector<std::complex<float>> inputs(N);
    std::vector<float> expectedOutputs(N);

    // Generate test data from example in VOLK header
    uint32_t jj = 0;
    const std::vector<float> constVals{-3, -1, 1, 3};
    for(uint32_t ii = 0; ii < N; ++ii)
    {
        inputs[ii] = {constVals[ii%4], constVals[jj]};
        if((ii+1)%4 == 0) ++jj;
    }
    for(uint32_t i = 0; i < N; ++i)
    {
        const auto diff = complexInput - inputs[i];
        expectedOutputs[i] = std::pow(diff.real(), 2.0f) + std::pow(diff.imag(), 2.0f);
    }

    std::cout << " * Testing with no scaling..." << std::endl;

    VOLKTests::testOneToOneBlock<std::complex<float>,float>(
        squareDist,
        inputs,
        expectedOutputs);

    std::cout << " * Testing with scaling..." << std::endl;

    constexpr float scalar = 10.0f;
    for(auto& expectedOutput: expectedOutputs) expectedOutput *= scalar;

    squareDist.call("setScalar", scalar);
    POTHOS_TEST_CLOSE(scalar, squareDist.call<float>("scalar"), 1e-6f);

    VOLKTests::testOneToOneBlock<std::complex<float>,float>(
        squareDist,
        inputs,
        expectedOutputs);
}

//
// /volk/sqrt
//

POTHOS_TEST_BLOCK("/volk/tests", test_sqrt)
{
    VOLKTests::testOneToOneBlock<float,float>(
        Pothos::BlockRegistry::make("/volk/sqrt"),
        {0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f},
        {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,  5.0f});
}

//
// /volk/subtract
//

POTHOS_TEST_BLOCK("/volk/tests", test_subtract)
{
    VOLKTests::testTwoToOneBlock<float,float,float,size_t>(
        Pothos::BlockRegistry::make("/volk/subtract"),
        {0.5f,  1.0f,  1.5f,  2.0f,  2.5f},
        {-1.0f, 1.5f,  -2.0f, 2.5f,  -3.0f},
        {1.5f,  -0.5f, 3.5f,  -0.5f, 5.5f},
        0,
        1);
}

//
// /volk/tan
//

POTHOS_TEST_BLOCK("/volk/tests", test_tan)
{
    VOLKTests::testOneToOneBlock<float,float>(
        Pothos::BlockRegistry::make("/volk/tan"),
        {0.0f, M_PI_4, M_PI},
        {0.0f, 1.0f,   0.0f});
}

//
// /volk/tanh
//

POTHOS_TEST_BLOCK("/volk/tests", test_tanh)
{
    VOLKTests::testOneToOneBlock<float,float>(
        Pothos::BlockRegistry::make("/volk/tanh"),
        {0.0f, M_PI_2,   M_PI},
        {0.0f, 0.91715f, 0.99627f});
}
