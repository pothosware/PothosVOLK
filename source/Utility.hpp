// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Pothos/Exception.hpp>
#include <Pothos/Framework.hpp>
#include <Pothos/Object.hpp>

#include <Poco/Format.h>

#include <algorithm>
#include <string>
#include <vector>

template <typename T>
static bool doesDTypeMatch(const Pothos::DType& dtype)
{
    static const auto DTypeT = Pothos::DType(typeid(T));

    return (Pothos::DType::fromDType(dtype, 1) == DTypeT);
}


template <typename T>
static std::string valueToString(const T& input)
{
    return Pothos::Object(input).toString();
}

template <typename T>
static std::string valueToString(const std::vector<T>& inputs)
{
    auto stringVectorObj = Pothos::Object::make<std::vector<std::string>>({});

    std::vector<std::string> strings;
    std::transform(
        inputs.begin(),
        inputs.end(),
        std::back_inserter(stringVectorObj.ref<std::vector<std::string>>()),
        [](const T& input){return Pothos::Object(input).toString();});

    return stringVectorObj.toString();
}

class InvalidDTypeException: public Pothos::InvalidArgumentException
{
    public:
        template <typename T>
        InvalidDTypeException(
            const std::string& context,
            const T& dtypes
        ):
            Pothos::InvalidArgumentException(Poco::format(
                "%s dtype(s): %s",
                context,
                valueToString(dtypes)))
        {}

        template <typename T1, typename T2>
        InvalidDTypeException(
            const std::string& context,
            const T1& dtypesIn,
            const T2& dtypesOut
        ):
            Pothos::InvalidArgumentException(Poco::format(
                "%s dtypes: %s -> %s",
                context,
                valueToString(dtypesIn),
                valueToString(dtypesOut)))
        {}

        template <typename T1, typename T2, typename T3>
        InvalidDTypeException(
            const std::string& context,
            const T1& dtypesIn,
            const T2& dtypesOut,
            const T3& paramDTypes
        ):
            Pothos::InvalidArgumentException(Poco::format(
                "%s dtypes: %s in, %s out, %s param(s)",
                context,
                valueToString(dtypesIn),
                valueToString(dtypesOut),
                valueToString(paramDTypes)))
        {}

        virtual ~InvalidDTypeException() = default;
};
