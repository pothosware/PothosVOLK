// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Utility.hpp"
#include "VOLKBlock.hpp"

#include <volk/volk.h>

#include <cassert>
#include <complex>
#include <string>
#include <vector>

#warning TODO: versions with scalar params

#define IfTypesThenOneToOneBlock(InType,OutType,fcn) \
    if(doesDTypeMatch<InType>(inDType) && doesDTypeMatch<OutType>(outDType)) \
        return OneToOneBlock<InType, OutType>::make(fcn);

#define IfTypesThenOneToTwoBlock(InType,OutType,fcn) \
    if(doesDTypeMatch<InType>(inDType) && doesDTypeMatch<OutType>(outDType)) \
        return OneToTwoBlock<InType, OutType, OutType>::make(fcn);

#define IfTypeThenTwoToOneBlock(Type,fcn) \
    if(doesDTypeMatch<Type>(dtype)) return TwoToOneBlock<Type, Type, Type>::make(fcn);

#define IfTypesThenTwoToOneBlock(InType0,InType1,OutType,fcn) \
    if(doesDTypeMatch<InType0>(inDType0) && doesDTypeMatch<InType1>(inDType1) && doesDTypeMatch<OutType>(outDType)) \
        return TwoToOneBlock<InType0, InType1, OutType>::make(fcn);

//
// /volk/acos
//

static const std::string VOLKACosPath = "/volk/acos";

static Pothos::BlockRegistry registerVOLKACos(
    VOLKACosPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_acos_32f, 0));

//
// /volk/asin
//

static const std::string VOLKASinPath = "/volk/asin";

static Pothos::BlockRegistry registerVOLKASin(
    VOLKASinPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_asin_32f, 0));

//
// /volk/atan
//

static const std::string VOLKATanPath = "/volk/atan";

static Pothos::BlockRegistry registerVOLKATan(
    VOLKATanPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_atan_32f, 0));

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

    throw InvalidDTypeException(VOLKAddPath, {inDType0, inDType1, outDType});
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
// /volk/binary_slicer
//

static const std::string VOLKBinarySlicerPath = "/volk/binary_slicer";

static Pothos::Block* makeBinarySlicer(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType)
{
    IfTypesThenOneToOneBlock(float,int8_t,volk_32f_binary_slicer_8i)
    IfTypesThenOneToOneBlock(float,int32_t,volk_32f_binary_slicer_32i)

    throw InvalidDTypeException(VOLKBinarySlicerPath, {inDType, outDType});
}

static Pothos::BlockRegistry registerVOLKBinarySlicer(
    VOLKBinarySlicerPath,
    &makeBinarySlicer);

//
// /volk/conjugate
//

static const std::string VOLKConjugatePath = "/volk/conjugate";

static Pothos::BlockRegistry registerVOLKConjugate(
    VOLKConjugatePath,
    Pothos::Callable(OneToOneBlock<std::complex<float>,std::complex<float>>::make)
        .bind(volk_32fc_conjugate_32fc, 0));

//
// /volk/convert
//

static const std::string VOLKConvertPath = "/volk/convert";

static Pothos::Block* makeConvert(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType)
{
    IfTypesThenOneToOneBlock(int8_t,int16_t,volk_8i_convert_16i)
    IfTypesThenOneToOneBlock(int16_t,int8_t,volk_16i_convert_8i)
    IfTypesThenOneToOneBlock(float,double,volk_32f_convert_64f)
    IfTypesThenOneToOneBlock(double,float,volk_64f_convert_32f)
    IfTypesThenOneToOneBlock(std::complex<int16_t>,std::complex<float>,volk_16ic_convert_32fc)
    IfTypesThenOneToOneBlock(std::complex<float>,std::complex<int16_t>,volk_32fc_convert_16ic)

    throw InvalidDTypeException(VOLKConvertPath, {inDType, outDType});
}

static Pothos::BlockRegistry registerVOLKConvert(
    VOLKConvertPath,
    &makeConvert);

//
// /volk/cos
//

static const std::string VOLKCosPath = "/volk/cos";

static Pothos::BlockRegistry registerVOLKCos(
    VOLKCosPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_cos_32f, 0));

//
// /volk/deinterleave
//

static const std::string VOLKDeinterleavePath = "/volk/deinterleave";

static Pothos::Block* makeDeinterleave(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType)
{
    IfTypesThenOneToTwoBlock(std::complex<int8_t>,int16_t,volk_8ic_deinterleave_16i_x2)
    IfTypesThenOneToTwoBlock(std::complex<int16_t>,int16_t,volk_16ic_deinterleave_16i_x2)
    IfTypesThenOneToTwoBlock(std::complex<float>,float,volk_32fc_deinterleave_32f_x2)
    IfTypesThenOneToTwoBlock(std::complex<float>,double,volk_32fc_deinterleave_64f_x2)

    throw InvalidDTypeException(VOLKDeinterleavePath, {inDType, outDType});
}

static Pothos::BlockRegistry registerVOLKDeinterleave(
    VOLKDeinterleavePath,
    &makeDeinterleave);

//
// /volk/deinterleave_imag
//

static const std::string VOLKDeinterleaveImagPath = "/volk/deinterleave_imag";

static Pothos::BlockRegistry registerVOLKDeinterleaveImag(
    VOLKDeinterleaveImagPath,
    Pothos::Callable(OneToOneBlock<std::complex<float>,float>::make)
        .bind(volk_32fc_deinterleave_imag_32f, 0));

//
// /volk/deinterleave_real
//

static const std::string VOLKDeinterleaveRealPath = "/volk/deinterleave_real";

static Pothos::Block* makeDeinterleaveReal(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType)
{
    IfTypesThenOneToOneBlock(std::complex<int8_t>,int8_t,volk_8ic_deinterleave_real_8i)
    IfTypesThenOneToOneBlock(std::complex<int8_t>,int16_t,volk_8ic_deinterleave_real_16i)
    IfTypesThenOneToOneBlock(std::complex<int16_t>,int8_t,volk_16ic_deinterleave_real_8i)
    IfTypesThenOneToOneBlock(std::complex<int16_t>,int16_t,volk_16ic_deinterleave_real_16i)
    IfTypesThenOneToOneBlock(std::complex<float>,float,volk_32fc_deinterleave_real_32f)
    IfTypesThenOneToOneBlock(std::complex<float>,double,volk_32fc_deinterleave_real_64f)

    throw InvalidDTypeException(VOLKDeinterleaveRealPath, {inDType, outDType});
}

static Pothos::BlockRegistry registerVOLKDeinterleaveReal(
    VOLKDeinterleaveRealPath,
    &makeDeinterleaveReal);

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

    throw InvalidDTypeException(VOLKDividePath, {inDType0, inDType1, outDType});
}

static Pothos::BlockRegistry registerVOLKDivide(
    VOLKDividePath,
    &makeDivide);

//
// /volk/exp
//

static const std::string VOLKExpPath = "/volk/exp";

static Pothos::Block* makeExp(const std::string& mode)
{
    OneToOneFcn<float,float> volkFcn = nullptr;

    if(mode == "PRECISE")   volkFcn = ::volk_32f_exp_32f;
    else if(mode == "FAST") volkFcn = ::volk_32f_expfast_32f;
    else throw Pothos::InvalidArgumentException(VOLKExpPath + " mode: " + mode);

    assert(volkFcn);

    return OneToOneBlock<float,float>::make(volkFcn);
}

static Pothos::BlockRegistry registerVOLKExp(
    VOLKExpPath,
    makeExp);

//
// /volk/interleave
//

static const std::string VOLKInterleavePath = "/volk/interleave";

static Pothos::BlockRegistry registerVOLKInterleave(
    VOLKInterleavePath,
    Pothos::Callable(TwoToOneBlock<float,float,std::complex<float>>::make)
        .bind(volk_32f_x2_interleave_32fc, 0));

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
// /volk/max
//

static const std::string VOLKMaxPath = "/volk/max";

static Pothos::Block* makeMax(const Pothos::DType& dtype)
{
    IfTypeThenTwoToOneBlock(float,volk_32f_x2_max_32f)
    IfTypeThenTwoToOneBlock(double,volk_64f_x2_max_64f)

    throw InvalidDTypeException(VOLKMaxPath, dtype);
}

static Pothos::BlockRegistry registerVOLKMax(
    VOLKMaxPath,
    &makeMax);

//
// /volk/min
//

static const std::string VOLKMinPath = "/volk/min";

static Pothos::Block* makeMin(const Pothos::DType& dtype)
{
    IfTypeThenTwoToOneBlock(float,volk_32f_x2_min_32f)
    IfTypeThenTwoToOneBlock(double,volk_64f_x2_min_64f)

    throw InvalidDTypeException(VOLKMinPath, dtype);
}

static Pothos::BlockRegistry registerVOLKMin(
    VOLKMinPath,
    &makeMin);

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

    throw InvalidDTypeException(VOLKMultiplyPath, {inDType0, inDType1, outDType});
}

static Pothos::BlockRegistry registerVOLKMultiply(
    VOLKMultiplyPath,
    &makeMultiply);

//
// /volk/multiply_conjugate
//

static const std::string VOLKMultiplyConjugatePath = "/volk/multiply_conjugate";

static Pothos::Block* makeMultiplyConjugate(
    const Pothos::DType& inDType0,
    const Pothos::DType& inDType1,
    const Pothos::DType& outDType)
{
    IfTypesThenTwoToOneBlock(std::complex<int8_t>,std::complex<int8_t>,std::complex<int16_t>,volk_8ic_x2_multiply_conjugate_16ic)
    IfTypesThenTwoToOneBlock(std::complex<float>,std::complex<float>,std::complex<float>,volk_32fc_x2_multiply_conjugate_32fc)

    throw InvalidDTypeException(VOLKMultiplyConjugatePath, {inDType0, inDType1, outDType});
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
// /volk/pow
//

static const std::string VOLKPowPath = "/volk/pow";

static Pothos::BlockRegistry registerVOLKPow(
    VOLKPowPath,
    Pothos::Callable(TwoToOneBlock<float,float,float>::make)
        .bind(volk_32f_x2_pow_32f, 0));

//
// /volk/reverse
//

static const std::string VOLKReversePath = "/volk/reverse";

static Pothos::BlockRegistry registerVOLKReverse(
    VOLKReversePath,
    Pothos::Callable(OneToOneBlock<uint32_t,uint32_t>::make)
        .bind(volk_32u_reverse_32u, 0));

//
// /volk/sin
//

static const std::string VOLKSinPath = "/volk/sin";

static Pothos::BlockRegistry registerVOLKSin(
    VOLKSinPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_sin_32f, 0));

//
// /volk/square_dist
//

static const std::string VOLKSquareDistPath = "/volk/square_dist";

static Pothos::BlockRegistry registerVOLKSquareDist(
    VOLKSquareDistPath,
    Pothos::Callable(TwoToOneBlock<std::complex<float>,std::complex<float>,float>::make)
        .bind(volk_32fc_x2_square_dist_32f, 0));

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

//
// /volk/tan
//

static const std::string VOLKTanPath = "/volk/tan";

static Pothos::BlockRegistry registerVOLKTan(
    VOLKTanPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_tan_32f, 0));

//
// /volk/tanh
//

static const std::string VOLKTanHPath = "/volk/tanh";

static Pothos::BlockRegistry registerVOLKTanH(
    VOLKTanHPath,
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_tanh_32f, 0));
