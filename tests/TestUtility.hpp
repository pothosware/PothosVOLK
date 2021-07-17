// Copyright (c) 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Pothos/Framework.hpp>
#include <Pothos/Testing.hpp>

#include <complex>
#include <cstring>
#include <type_traits>
#include <vector>

namespace VOLKTests
{
    template <typename T>
    struct IsComplex : std::false_type {};

    template <typename T>
    struct IsComplex<std::complex<T>> : std::true_type {};

    template <typename T, typename Ret>
    using EnableIfIntegral = typename std::enable_if<std::is_integral<T>::value && !IsComplex<T>::value, Ret>::type;

    template <typename T, typename Ret>
    using DisableIfIntegral = typename std::enable_if<!std::is_integral<T>::value && !IsComplex<T>::value, Ret>::type;

    template <typename T, typename Ret>
    using EnableIfComplex = typename std::enable_if<IsComplex<T>::value, Ret>::type;

    template <typename T, typename Ret>
    using DisableIfComplex = typename std::enable_if<!IsComplex<T>::value, Ret>::type;

    template <typename T>
    static constexpr T epsilon() {return T(0);}

    template <>
    constexpr float epsilon<float>() {return 1e-6f;}

    template <>
    constexpr double epsilon<double>() {return 1e-6;}

    template <typename T>
    static Pothos::BufferChunk stdVectorToBufferChunk(const std::vector<T>& inputs)
    {
        Pothos::BufferChunk ret(Pothos::DType(typeid(T)), inputs.size());
        std::memcpy(
            reinterpret_cast<void*>(ret.address),
            inputs.data(),
            ret.length);

        return ret;
    }

    // Copy the input vector some number of times and return a longer vector. This
    // is used to make sure when SIMD implementations are used, the test data is
    // long enough that the SIMD codepaths are tested.
    template <typename T>
    static std::vector<T> stretchStdVector(
        const std::vector<T>& inputs,
        size_t numRepetitions)
    {
        std::vector<T> outputs;
        outputs.reserve(inputs.size() * numRepetitions);

        for(size_t i = 0; i < numRepetitions; ++i)
        {
            outputs.insert(outputs.end(), inputs.begin(), inputs.end());
        }

        return outputs;
    }

    template <typename T>
    static inline Pothos::BufferChunk stdVectorToStretchedBufferChunk(
        const std::vector<T>& inputs,
        size_t numRepetitions)
    {
        return stdVectorToBufferChunk<T>(stretchStdVector<T>(inputs, numRepetitions));
    }

    template <typename T>
    static EnableIfIntegral<T, void> testValuesEqual(
        const T& expected,
        const T& actual)
    {
        POTHOS_TEST_EQUAL(expected, actual);
    }

    template <typename T>
    static DisableIfIntegral<T, void> testValuesEqual(
        const T& expected,
        const T& actual)
    {
        POTHOS_TEST_CLOSE(expected, actual, epsilon<T>());
    }

    template <typename T>
    static EnableIfComplex<T, void> testValuesEqual(
        const T& expected,
        const T& actual)
    {
        testValuesEqual(expected.real(), actual.real());
        testValuesEqual(expected.imag(), actual.imag());
    }

    template <typename T>
    static void testBufferChunksEqual(
        const Pothos::BufferChunk& expected,
        const Pothos::BufferChunk& actual)
    {
        POTHOS_TEST_EQUAL(expected.dtype, actual.dtype);
        POTHOS_TEST_EQUAL(expected.elements(), actual.elements());
        POTHOS_TEST_EQUALA(
            expected.as<const T*>(),
            actual.as<const T*>(),
            expected.elements());
    }

    template <typename T>
    static void testBufferChunksClose(
        const Pothos::BufferChunk& expected,
        const Pothos::BufferChunk& actual,
        T epsilon)
    {
        POTHOS_TEST_EQUAL(expected.dtype, actual.dtype);
        POTHOS_TEST_EQUAL(expected.elements(), actual.elements());
        POTHOS_TEST_CLOSEA(
            expected.as<const T*>(),
            actual.as<const T*>(),
            epsilon,
            expected.elements());
    }
}
