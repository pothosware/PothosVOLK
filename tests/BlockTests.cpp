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
    std::vector<float> expectedOutputs = {M_PI_2, M_PI/3.0f, 0.0f};

    auto acos = Pothos::BlockRegistry::make("/volk/acos");

    VOLKTests::testOneToOneBlock<float,float>(
        acos,
        testInputs,
        expectedOutputs);
}

// TODO: /volk/accumulator

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

// TODO: add_quad
// TODO: add_scalar

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

// TODO: /volk/atan2

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

// TODO: /volk/calc_spectral_noise_floor

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

// TODO: /volk/convert_scaled

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

// TODO: /volk/deinterleave_real_scaled
// TODO: /volk/deinterleave_scaled

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

static void testExp(const std::string& mode, bool laxEpsilon)
{
    std::cout << "Testing " << mode << " mode..." << std::endl;

    VOLKTests::testOneToOneBlock<float,float>(
        Pothos::BlockRegistry::make("/volk/exp", mode),
        {0.0f, 1.0f},
        {1.0f, M_E},
        laxEpsilon);
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

// TODO: /volk/interleave_scaled

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

// TODO: /volk/max_star

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

// TODO: /volk/multiply_conjugate_add
// TODO: /volk/multiply_conjugate_scaled
// TODO: /volk/multiply_scalar
// TODO: /volk/normalize

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

// TODO: /volk/popcnt

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

// TODO: /volk/power
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
