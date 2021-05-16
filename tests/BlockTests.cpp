// Copyright (c) 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "BlockTests.hpp"
#include "TestUtility.hpp"

#include <Pothos/Framework.hpp>
#include <Pothos/Testing.hpp>

#include <cmath>
#include <iostream>
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
