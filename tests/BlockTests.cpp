// Copyright (c) 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "BlockTests.hpp"
#include "TestUtility.hpp"

#include <Pothos/Framework.hpp>
#include <Pothos/Testing.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>

//
// /volk/acos
//

POTHOS_TEST_BLOCK("/volk/tests", test_acos)
{
    std::vector<float> testInputs      = {0.0f, 0.5f, 1.0f};
    std::vector<float> expectedOutputs = {M_PI/2.0f, M_PI/3.0f, 0.0f};

    auto acos = Pothos::BlockRegistry::make("/volk/acos");

    VOLKTests::testOneToOneBlock<float,float>(
        acos,
        testInputs,
        expectedOutputs);
}

//
// /volk/asin
//

POTHOS_TEST_BLOCK("/volk/tests", test_asin)
{
    std::vector<float> testInputs      = {0.0f, 0.5f, 1.0f};
    std::vector<float> expectedOutputs = {0.0f, M_PI/6.0f, M_PI/2.0f};

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
    std::vector<float> expectedOutputs = {0.0f, M_PI/4.0f, M_PI/2.0f};

    auto atan = Pothos::BlockRegistry::make("/volk/atan");

    VOLKTests::testOneToOneBlock<float,float>(
        atan,
        testInputs,
        expectedOutputs);
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

    VOLKTests::testTwoToOneBlock<InType0,InType1,OutType>(
        add,
        testInputs0,
        testInputs1,
        expectedOutputs);
}

POTHOS_TEST_BLOCK("/volk/tests", test_add)
{
    testAdd<float,float,float>();
    testAdd<float,double,double>();
    testAdd<double,double,double>();
    testAdd<std::complex<float>,std::complex<float>,std::complex<float>>();
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

    VOLKTests::testTwoToOneBlock<int,int,int>(
        andBlock,
        testInputs0,
        testInputs1,
        expectedOutputs);
}

//
// /volk/binary_slicer
//

template <typename OutputType>
static void testBinarySlicer()
{
    const auto outputDType = Pothos::DType(typeid(OutputType));

    std::cout << " * Testing float32 -> " << outputDType.name()
              << "..." << std::endl;

    std::vector<float> inputs               = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
    std::vector<OutputType> expectedOutputs = {0, 0, 1, 1, 1};

    auto binarySlicerBlock = Pothos::BlockRegistry::make(
        "/volk/binary_slicer",
        "float32",
        outputDType);

    VOLKTests::testOneToOneBlock<float,OutputType>(
        binarySlicerBlock,
        inputs,
        expectedOutputs);
}

POTHOS_TEST_BLOCK("/volk/tests", test_binary_slicer)
{
    testBinarySlicer<int8_t>();
    testBinarySlicer<int32_t>();
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
