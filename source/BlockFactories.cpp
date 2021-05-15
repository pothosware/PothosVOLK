// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "VOLKBlock.hpp"

#include <Pothos/Exception.hpp>
#include <Pothos/Object.hpp>

#include <Poco/Format.h>

#include <volk/volk.h>

#include <complex>
#include <string>
#include <vector>

//
// Utility
//

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

#define IfTypeThenTwoToOneBlock(Type,fcn) \
    if(doesDTypeMatch<Type>(dtype)) return TwoToOneBlock<Type, Type, Type>::make(fcn);

#define IfTypesThenTwoToOneBlock(InType0,InType1,OutType,fcn) \
    if(doesDTypeMatch<InType0>(inDType0) && doesDTypeMatch<InType1>(inDType1) && doesDTypeMatch<OutType>(outDType)) \
        return TwoToOneBlock<InType0, InType1, OutType>::make(fcn);

//
// /volk/add
//

static const std::string VOLKAddPath = "/volk/add";

static Pothos::Block* makeAdd(
    const Pothos::DType& inDType0,
    const Pothos::DType& inDType1,
    const Pothos::DType& outDType)
{
    IfTypesThenTwoToOneBlock(float,float,float,volk_32f_x2_add_32f)
    IfTypesThenTwoToOneBlock(float,double,double,volk_32f_64f_add_64f)
    IfTypesThenTwoToOneBlock(double,double,double,volk_64f_x2_add_64f)
    IfTypesThenTwoToOneBlock(std::complex<float>,std::complex<float>,std::complex<float>,volk_32fc_x2_add_32fc)

    throw InvalidDTypesException(VOLKAddPath, {inDType0, inDType1, outDType});
}

static Pothos::BlockRegistry registerVOLKAdd(
    VOLKAddPath,
    &makeAdd);

//
// /volk/and
//

static const std::string VOLKAndPath = "/volk/and";

static Pothos::BlockRegistry registerVOLKAnd(
    VOLKAndPath,
    Pothos::Callable(TwoToOneBlock<int,int,int>::make)
        .bind(volk_32i_x2_and_32i, 0));

//
// /volk/conjugate
//

static const std::string VOLKConjugatePath = "/volk/conjugate";

static Pothos::BlockRegistry registerVOLKConjugate(
    VOLKConjugatePath,
    Pothos::Callable(OneToOneBlock<std::complex<float>,std::complex<float>>::make)
        .bind(volk_32fc_conjugate_32fc, 0));

//
// /volk/divide
//

static const std::string VOLKDividePath = "/volk/divide";

static Pothos::Block* makeDivide(
    const Pothos::DType& inDType0,
    const Pothos::DType& inDType1,
    const Pothos::DType& outDType)
{
    IfTypesThenTwoToOneBlock(float,float,float,volk_32f_x2_divide_32f)
    IfTypesThenTwoToOneBlock(std::complex<float>,std::complex<float>,std::complex<float>,volk_32fc_x2_divide_32fc)

    throw InvalidDTypesException(VOLKDividePath, {inDType0, inDType1, outDType});
}

static Pothos::BlockRegistry registerVOLKDivide(
    VOLKDividePath,
    &makeDivide);

//
// /volk/exp
//

static const std::string VOLKExpPath = "/volk/exp";

static Pothos::BlockRegistry registerVOLKExp(
    VOLKExpPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_exp_32f, 0));

//
// /volk/expfast
//

static const std::string VOLKExpFastPath = "/volk/expfast";

static Pothos::BlockRegistry registerVOLKexpfastFast(
    VOLKExpFastPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_expfast_32f, 0));

//
// /volk/invsqrt
//

static const std::string VOLKInvSqrtPath = "/volk/invsqrt";

static Pothos::BlockRegistry registerVOLKInvSqrt(
    VOLKInvSqrtPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_invsqrt_32f, 0));

//
// /volk/log2
//

static const std::string VOLKLog2Path = "/volk/log2";

static Pothos::BlockRegistry registerVOLKLog2(
    VOLKLog2Path,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_log2_32f, 0));

//
// /volk/min
//

static const std::string VOLKMinPath = "/volk/min";

static Pothos::Block* makeMin(const Pothos::DType& dtype)
{
    IfTypeThenTwoToOneBlock(float,volk_32f_x2_min_32f)
    IfTypeThenTwoToOneBlock(double,volk_64f_x2_min_64f)

    throw InvalidDTypesException(VOLKMinPath, {dtype});
}

static Pothos::BlockRegistry registerVOLKMin(
    VOLKMinPath,
    &makeMin);

//
// /volk/max
//

static const std::string VOLKMaxPath = "/volk/max";

static Pothos::Block* makeMax(const Pothos::DType& dtype)
{
    IfTypeThenTwoToOneBlock(float,volk_32f_x2_max_32f)
    IfTypeThenTwoToOneBlock(double,volk_64f_x2_max_64f)

    throw InvalidDTypesException(VOLKMaxPath, {dtype});
}

static Pothos::BlockRegistry registerVOLKMax(
    VOLKMaxPath,
    &makeMax);

//
// /volk/multiply
//

static const std::string VOLKMultiplyPath = "/volk/multiply";

static Pothos::Block* makeMultiply(
    const Pothos::DType& inDType0,
    const Pothos::DType& inDType1,
    const Pothos::DType& outDType)
{
    IfTypesThenTwoToOneBlock(float,double,double,volk_32f_64f_multiply_64f)
    IfTypesThenTwoToOneBlock(double,double,double,volk_64f_x2_multiply_64f)
    IfTypesThenTwoToOneBlock(std::complex<int16_t>,std::complex<int16_t>,std::complex<int16_t>,volk_16ic_x2_multiply_16ic)
    IfTypesThenTwoToOneBlock(std::complex<float>,std::complex<float>,std::complex<float>,volk_32fc_x2_multiply_32fc)
    IfTypesThenTwoToOneBlock(std::complex<float>,float,std::complex<float>,volk_32fc_32f_multiply_32fc)

    throw InvalidDTypesException(VOLKMultiplyPath, {inDType0, inDType1, outDType});
}

static Pothos::BlockRegistry registerVOLKMultiply(
    VOLKMultiplyPath,
    &makeMultiply);

//
// /volk/multiply
//

static const std::string VOLKMultiplyConjugatePath = "/volk/multiply";

static Pothos::Block* makeMultiplyConjugate(
    const Pothos::DType& inDType0,
    const Pothos::DType& inDType1,
    const Pothos::DType& outDType)
{
    IfTypesThenTwoToOneBlock(std::complex<int8_t>,std::complex<int8_t>,std::complex<int16_t>,volk_8ic_x2_multiply_conjugate_16ic)
    IfTypesThenTwoToOneBlock(std::complex<float>,std::complex<float>,std::complex<float>,volk_32fc_x2_multiply_conjugate_32fc)

    throw InvalidDTypesException(VOLKMultiplyConjugatePath, {inDType0, inDType1, outDType});
}

static Pothos::BlockRegistry registerVOLKMultiplyConjugate(
    VOLKMultiplyConjugatePath,
    &makeMultiplyConjugate);

//
// /volk/or
//

static const std::string VOLKOrPath = "/volk/or";

static Pothos::BlockRegistry registerVOLKOr(
    VOLKOrPath,
    Pothos::Callable(TwoToOneBlock<int,int,int>::make)
        .bind(volk_32i_x2_or_32i, 0));

//
// /volk/sqrt
//

static const std::string VOLKSqrtPath = "/volk/sqrt";

static Pothos::BlockRegistry registerVOLKSqrt(
    VOLKSqrtPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_sqrt_32f, 0));

//
// /volk/subtract
//

static const std::string VOLKSubtractPath = "/volk/subtract";

static Pothos::BlockRegistry registerVOLKSubtract(
    VOLKSubtractPath,
    Pothos::Callable(TwoToOneBlock<float,float,float>::make)
        .bind(volk_32f_x2_subtract_32f, 0));
