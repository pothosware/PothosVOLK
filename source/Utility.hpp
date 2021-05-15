// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Pothos/Framework.hpp>
#include <Pothos/Object.hpp>

#include <Poco/Format.h>

#include <vector>

template <typename T>
static bool doesDTypeMatch(const Pothos::DType& dtype)
{
    static const auto DTypeT = Pothos::DType(typeid(T));

    return (Pothos::DType::fromDType(dtype, 1) == DTypeT);
}

class InvalidDTypesException: public Pothos::InvalidArgumentException
{
    public:
        InvalidDTypesException(
            const std::string& context,
            const std::vector<Pothos::DType>& dtypes
        ):
            Pothos::InvalidArgumentException(Poco::format(
                "%s: %s",
                context,
                Pothos::Object(dtypes).toString()))
        {}

        virtual ~InvalidDTypesException() = default;
};
