// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Pothos/Exception.hpp>
#include <Pothos/Framework.hpp>
#include <Pothos/Object.hpp>

#include <Poco/Format.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

template <typename T>
static bool doesDTypeMatch(const Pothos::DType& dtype)
{
    static const auto DTypeT = Pothos::DType(typeid(T));

    return (Pothos::DType::fromDType(dtype, 1) == DTypeT);
}

inline std::vector<std::string> dtypesToNames(const std::vector<Pothos::DType>& dtypes)
{
    std::vector<std::string> names;
    std::transform(
        dtypes.begin(),
        dtypes.end(),
        std::back_inserter(names),
        std::mem_fn(&Pothos::DType::toString));

    return names;
}

// TODO: separate input and output types in message
class InvalidDTypeException: public Pothos::InvalidArgumentException
{
    public:
        InvalidDTypeException(
            const std::string& context,
            const Pothos::DType& dtype
        ):
            Pothos::InvalidArgumentException(Poco::format(
                "%s dtypes: %s",
                context,
                dtype.toString()))
        {}
        InvalidDTypeException(
            const std::string& context,
            const std::vector<Pothos::DType>& dtypes
        ):
            Pothos::InvalidArgumentException(Poco::format(
                "%s dtypes: %s",
                context,
                Pothos::Object(dtypesToNames(dtypes)).toString()))
        {}

        virtual ~InvalidDTypeException() = default;
};
