// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Utility.hpp"
#include "VOLKBlock.hpp"

#include <volk/volk.h>

#include <cassert>
#include <complex>
#include <string>
#include <vector>

#warning TODO: consistency in future-proofing with type parameters

#define IfTypesThenOneToOneBlock(InType,OutType,fcn) \
    if(doesDTypeMatch<InType>(inDType) && doesDTypeMatch<OutType>(outDType)) \
        return OneToOneBlock<InType, OutType>::make(fcn);

#define IfTypesThenOneToOneScalarParamBlock(InType,OutType,ScalarType,GetterName,SetterName,Fcn) \
    if(doesDTypeMatch<InType>(inDType) && doesDTypeMatch<OutType>(outDType) && doesDTypeMatch<ScalarType>(scalarDType)) \
        return new OneToOneScalarParamBlock<InType,OutType,ScalarType>( \
            Fcn, \
            GetterName, \
            SetterName);

#define IfTypesThenOneToTwoBlock(InType,OutType,OutputPortType,Fcn,port0Name,port1Name) \
    if(doesDTypeMatch<InType>(inDType) && doesDTypeMatch<OutType>(outDType)) \
        return OneToTwoBlock<InType, OutType, OutType, OutputPortType>::make( \
            Fcn, \
            port0Name, \
            port1Name);

#define IfTypesThenOneToTwoScalarParamBlock(InType,OutType,ScalarType,OutputPortType,GetterName,SetterName,Fcn,port0Name,port1Name) \
    if(doesDTypeMatch<InType>(inDType) && doesDTypeMatch<OutType>(outDType) && doesDTypeMatch<ScalarType>(scalarDType)) \
        return new OneToTwoScalarParamBlock<InType,OutType,OutType,ScalarType,OutputPortType>( \
            Fcn, \
            GetterName, \
            SetterName, \
            port0Name, \
            port1Name);

#define IfTypeThenTwoToOneBlock(Type,InputPortType,fcn,port0Name,port1Name) \
    if(doesDTypeMatch<Type>(dtype)) return TwoToOneBlock<Type, Type, Type, InputPortType>::make(fcn, port0Name, port1Name);

#define IfTypesThenTwoToOneBlock(InType0,InType1,OutType,InputPortType,fcn,port0Name,port1Name) \
    if(doesDTypeMatch<InType0>(inDType0) && doesDTypeMatch<InType1>(inDType1) && doesDTypeMatch<OutType>(outDType)) \
        return TwoToOneBlock<InType0, InType1, OutType, InputPortType>::make(fcn,port0Name,port1Name);

#define IfTypesThenTwoToOneScalarParamBlock(InType0,InType1,OutType,ScalarType,GetterName,SetterName,Fcn) \
    if(doesDTypeMatch<InType0>(inDType0) && doesDTypeMatch<InType1>(inDType1) && doesDTypeMatch<OutType>(outDType) && doesDTypeMatch<ScalarType>(scalarDType)) \
        return new TwoToOneScalarParamBlock<InType0,InType1,OutType,ScalarType>( \
            Fcn, \
            GetterName, \
            SetterName);

//
// /volk/acos
//

/***********************************************************************
 * |PothosDoc ACos (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_acos_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/acos()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKACos(
    "/volk/acos",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_acos_32f, 0));

/***********************************************************************
 * |PothosDoc Add (VOLK)
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_32f_x2_add_32f</b></li>
 * <li><b>volk_32f_64f_add_64f</b></li>
 * <li><b>volk_64f_x2_add_64f</b></li>
 * <li><b>volk_32fc_32f_add_32fc</b></li>
 * <li><b>volk_32fc_x2_add_32fc</b></li>
 * </ul>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math plus
 *
 * |param input0DType[Data Type In0]
 * |widget DTypeChooser(float=1,cfloat32=1)
 * |default "float64"
 * |preview disable
 *
 * |param input1DType[Data Type In1]
 * |widget DTypeChooser(float=1,cfloat32=1)
 * |default "float64"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(float=1,cfloat32=1)
 * |default "float64"
 * |preview disable
 *
 * |factory /volk/add(input0DType,input1DType,outputDType)
 **********************************************************************/
static const std::string VOLKAddPath = "/volk/add";

static Pothos::Block* makeAdd(
    const Pothos::DType& inDType0,
    const Pothos::DType& inDType1,
    const Pothos::DType& outDType)
{
#define IfTypesThenAdd(in0,in1,out,fcn) \
    IfTypesThenTwoToOneBlock(in0,in1,out,size_t,fcn,0,1)

    IfTypesThenAdd(float,float,float,volk_32f_x2_add_32f)
    IfTypesThenAdd(float,double,double,volk_32f_64f_add_64f)
    IfTypesThenAdd(double,double,double,volk_64f_x2_add_64f)
    IfTypesThenAdd(std::complex<float>,std::complex<float>,std::complex<float>,volk_32fc_x2_add_32fc)
    IfTypesThenAdd(std::complex<float>,float,std::complex<float>,volk_32fc_32f_add_32fc)

    throw InvalidDTypeException(
        VOLKAddPath,
        std::vector<Pothos::DType>{inDType0, inDType1}, 
        outDType);
}

static Pothos::BlockRegistry registerVOLKAdd(
    VOLKAddPath,
    &makeAdd);

/***********************************************************************
 * |PothosDoc Scalar Add (VOLK)
 *
 * <p>
 * Adds a given scalar constant to all elements.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32f_s32f_add_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math plus constant
 *
 * |param scalar[Scalar] A constant value added to all inputs.
 * |widget DoubleSpinBox(decimals=3)
 * |default 0.0
 * |preview enable
 *
 * |factory /volk/add_scalar()
 * |setter setScalar(scalar)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKAddScalar(
    "/volk/add_scalar",
    Pothos::Callable(OneToOneScalarParamBlock<float,float,float>::make)
        .bind(volk_32f_s32f_add_32f, 0)
        .bind("scalar", 1)
        .bind("setScalar", 2));

/***********************************************************************
 * |PothosDoc Logical And (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32i_x2_and_32i</b>
 * </p>
 *
 * |category /Digital
 * |category /VOLK
 *
 * |factory /volk/and()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKAnd(
    "/volk/and",
    Pothos::Callable(TwoToOneBlock<int,int,int,size_t>::make)
        .bind(volk_32i_x2_and_32i, 0)
        .bind(0, 1)
        .bind(1, 2));

/***********************************************************************
 * |PothosDoc ASin (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_asin_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/asin()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKASin(
    "/volk/asin",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_asin_32f, 0));

/***********************************************************************
 * |PothosDoc ATan (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_atan_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/atan()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKATan(
    "/volk/atan",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_atan_32f, 0));

/***********************************************************************
 * |PothosDoc ATan2 (VOLK)
 *
 * <p>
 * Computes arctangent operation and applies a normalization factor.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32fc_s32f_atan2_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |param normalizationFactor[Normalization Factor]
 * A value multiplied to all <b>atan2</b> outputs.
 * |widget DoubleSpinBox(decimals=3)
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/atan2()
 * |setter setNormalizationFactor(normalizationFactor)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKATan2(
    "/volk/atan2",
    Pothos::Callable(OneToOneScalarParamBlock<std::complex<float>,float,float>::make)
        .bind(volk_32fc_s32f_atan2_32f, 0)
        .bind("normalizationFactor", 1)
        .bind("setNormalizationFactor", 2));

/***********************************************************************
 * |PothosDoc Binary Slicer (VOLK)
 *
 * <p>
 * For each element, outputs <b>1</b> if the value is <b>>= 0</b>
 * and <b>0</b> if the value is <b>< 0</b>.
 * </p>
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_32f_binary_slicer_8i</b></li>
 * <li><b>volk_32f_binary_slicer_32i</b></li>
 * </ul>
 *
 * |category /Stream
 * |category /VOLK
 * |keywords positive negative
 *
 * |param inputDType[Data Type In]
 * |widget DTypeChooser(float32=1)
 * |default "float32"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(int8=1,int32=1)
 * |default "int8"
 * |preview disable
 *
 * |factory /volk/binary_slicer(inputDType,outputDType)
 **********************************************************************/
static const std::string VOLKBinarySlicerPath = "/volk/binary_slicer";

static Pothos::Block* makeBinarySlicer(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType)
{
    IfTypesThenOneToOneBlock(float,int8_t,volk_32f_binary_slicer_8i)
    IfTypesThenOneToOneBlock(float,int32_t,volk_32f_binary_slicer_32i)

    throw InvalidDTypeException(
        VOLKBinarySlicerPath,
        inDType,
        outDType);
}

static Pothos::BlockRegistry registerVOLKBinarySlicer(
    VOLKBinarySlicerPath,
    &makeBinarySlicer);

/***********************************************************************
 * |PothosDoc Calc Spectral Noise Floor (VOLK)
 *
 * <p>Computes the spectral noise floor of an input power spectrum.</p>
 *
 * <p>
 * Calculates the spectral noise floor of an input power spectrum by
 * determining the mean of the input power spectrum, then
 * recalculating the mean excluding any power spectrum values that
 * exceed the mean by the <b>spectralExclusionValue</b> (in dB).  Provides a
 * rough estimation of the signal noise floor.
 * </p>
 *
 * <p>
 * Outputs the noise floor of the input spectrum in dB.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32f_s32f_calc_spectral_noise_floor_32f</b>
 * </p>
 *
 * |category /VOLK
 * |keywords rf spectrum
 *
 * |param spectralExclusionValue[Spectral Exclusion Value]
 * The number of dB above the noise floor that a data point must be to be
 * excluded from the noise floor calculation.
 * |widget DoubleSpinBox(decimals=3)
 * |units dB
 * |default 20.0
 * |preview enable
 *
 * |factory /volk/calc_spectral_noise_floor()
 * |setter setSpectralExclusionValue(spectralExclusionValue)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKCalcSpectralNoiseFloor(
    "/volk/calc_spectral_noise_floor",
    Pothos::Callable(OneToOneScalarParamBlock<float,float,float>::make)
        .bind(volk_32f_s32f_calc_spectral_noise_floor_32f, 0)
        .bind("spectralExclusionValue", 1)
        .bind("setSpectralExclusionValue", 2));

/***********************************************************************
 * |PothosDoc Conjugate (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32fc_conjugate_32fc</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 *
 * |factory /volk/conjugate()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKConjugate(
    "/volk/conjugate",
    Pothos::Callable(OneToOneBlock<std::complex<float>,std::complex<float>>::make)
        .bind(volk_32fc_conjugate_32fc, 0));

/***********************************************************************
 * |PothosDoc Convert (VOLK)
 *
 * <p>
 * Supported conversions:
 * </p>
 *
 * <ul>
 *   <li>
 *     int8 -> int16
 *     <ul>
 *       <li>Underlying function: <b>volk_8i_convert_16i</b></li>
 *       <li>Multiplies all inputs by <b>256</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     int16 -> int8
 *     <ul>
 *       <li>Underlying function: <b>volk_16i_convert_8i</b></li>
 *       <li>Divides all inputs by <b>256</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     float32 -> float64
 *     <ul>
 *       <li>Underlying function: <b>volk_32f_convert_64f</b></li>
 *     </ul>
 *   </li>
 *   <li>
 *     float64 -> float32
 *     <ul>
 *       <li>Underlying function: <b>volk_64f_convert_32f</b></li>
 *     </ul>
 *   </li>
 *   <li>
 *     cint16 -> cfloat32
 *     <ul>
 *       <li>Underlying function: <b>volk_16ic_convert_32fc</b></li>
 *     </ul>
 *   </li>
 *   <li>
 *     cfloat32 -> cint16
 *     <ul>
 *       <li>Underlying function: <b>volk_32fc_convert_16ic</b></li>
 *       <li>Truncates all values to fit inside an <b>int16</b>.</li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * |category /Convert
 * |category /VOLK
 * |keywords type
 *
 * |param inputDType[Data Type In]
 * |widget DTypeChooser(int8=1,int16=1,float=1,cint16=1,cfloat32=1)
 * |default "float32"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(int8=1,int16=1,float=1,cint16=1,cfloat32=1)
 * |default "float64"
 * |preview disable
 *
 * |factory /volk/convert(inputDType,outputDType)
 **********************************************************************/
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

    throw InvalidDTypeException(
        VOLKConvertPath,
        inDType,
        outDType);
}

static Pothos::BlockRegistry registerVOLKConvert(
    VOLKConvertPath,
    &makeConvert);

/***********************************************************************
 * |PothosDoc Convert (Custom Scalar) (VOLK)
 *
 * <p>
 * Converts all values and applies a given scalar. Whether the scalar is
 * multiplied or divided depends on the conversion and is listed below.
 * </p>
 *
 * <p>
 * Supported conversions:
 * </p>
 *
 * <ul>
 *   <li>
 *     float32 -> int8 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_f32_sf32_convert_8i</b></li>
 *       <li>Multiplies all inputs by <b>scalar</b>.</li>
 *       <li>Truncates all scaled values to fit inside an <b>int8</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     float32 -> int16 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_f32_sf32_convert_16i</b></li>
 *       <li>Multiplies all inputs by <b>scalar</b>.</li>
 *       <li>Truncates all scaled values to fit inside an <b>int16</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     float32 -> int32 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_32i_s32f_convert_f32</b></li>
 *       <li>Divides all inputs by <b>scalar</b>.</li>
 *       <li>Truncates all scaled values to fit inside an <b>int32</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     int8 -> float32 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_8i_s32f_convert_f32</b></li>
 *       <li>Divides all inputs by <b>scalar</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     int16 -> float32 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_16i_s32f_convert_f32</b></li>
 *       <li>Divides all inputs by <b>scalar</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     int32 -> float32 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_32i_s32f_convert_f32</b></li>
 *       <li>Divides all inputs by <b>scalar</b>.</li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * |category /Convert
 * |category /VOLK
 * |keywords type
 *
 * |param inputDType[Data Type In]
 * |widget DTypeChooser(int8=1,int16=1,int32=1,float32=1)
 * |default "int32"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(int8=1,int16=1,int32=1,float32=1)
 * |default "float32"
 * |preview disable
 *
 * |param scalarDType[Scalar Data Type]
 * |widget DTypeChooser(float32=1)
 * |default "float32"
 * |preview disable
 *
 * |param scalar[Scalar] A scalar to apply to each input post-conversion.
 * |widget LineEdit()
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/convert_scaled(inputDType,outputDType,scalarDType)
 * |setter setScalar(scalar)
 **********************************************************************/
static const std::string VOLKConvertScaledPath = "/volk/convert_scaled";

static Pothos::Block* makeConvertScaled(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType,
    const Pothos::DType& scalarDType)
{
    #define IfTypesThenConvertScaledBlock(InType,OutType,ScalarType,Fcn) \
        IfTypesThenOneToOneScalarParamBlock(InType,OutType,ScalarType,"scalar","setScalar",Fcn)

    IfTypesThenConvertScaledBlock(float,int8_t,float,volk_32f_s32f_convert_8i)
    IfTypesThenConvertScaledBlock(float,int16_t,float,volk_32f_s32f_convert_16i)
    IfTypesThenConvertScaledBlock(float,int32_t,float,volk_32f_s32f_convert_32i)
    IfTypesThenConvertScaledBlock(int8_t,float,float,volk_8i_s32f_convert_32f)
    IfTypesThenConvertScaledBlock(int16_t,float,float,volk_16i_s32f_convert_32f)
    IfTypesThenConvertScaledBlock(int32_t,float,float,volk_32i_s32f_convert_32f)

    throw InvalidDTypeException(
        VOLKConvertPath,
        inDType,
        outDType,
        scalarDType);
}

static Pothos::BlockRegistry registerVOLKConvertScaled(
    VOLKConvertScaledPath,
    &makeConvertScaled);

/***********************************************************************
 * |PothosDoc Cos (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_cos_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/cos()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKCos(
    "/volk/cos",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_cos_32f, 0));

/***********************************************************************
 * |PothosDoc Deinterleave (VOLK)
 *
 * <p>
 * Deinterleaves a complex input into its real and imaginary inputs,
 * performing type conversions if needed.
 * </p>
 *
 * <p>
 * Supported types:
 * </p>
 *
 * <ul>
 *   <li>
 *     cint8 -> int16,int16
 *     <ul>
 *       <li>Underlying function: <b>volk_8ic_deinterleave_16i_x2</b></li>
 *       <li>Multiplies all output values by <b>256</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     cint16 -> int16,int16
 *     <ul>
 *       <li>Underlying function: <b>volk_16ic_deinterleave_16i_x2</b></li>
 *     </ul>
 *   </li>
 *   <li>
 *     cfloat32 -> float32,float32
 *     <ul>
 *       <li>Underlying function: <b>volk_32fc_deinterleave_32f_x2</b></li>
 *     </ul>
 *   </li>
 *   <li>
 *     cfloat32 -> float64,float64
 *     <ul>
 *       <li>Underlying function: <b>volk_32fc_deinterleave_64f_x2</b></li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * |category /Convert
 * |category /Stream
 * |category /VOLK
 * |keywords complex real imag
 *
 * |param inputDType[Data Type In]
 * |widget DTypeChooser(cint8=1,cint16=1,cfloat32=1)
 * |default "complex_float32"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(int16=1,float32=1,float64=1)
 * |default "float32"
 * |preview disable
 *
 * |factory /volk/deinterleave(inputDType,outputDType)
 **********************************************************************/
static const std::string VOLKDeinterleavePath = "/volk/deinterleave";

static Pothos::Block* makeDeinterleave(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType)
{
#define IfTypesThenDeinterleave(InType,OutType,Fcn) \
    IfTypesThenOneToTwoBlock(InType,OutType,std::string,Fcn,"real","imag")

    IfTypesThenDeinterleave(std::complex<int8_t>,int16_t,volk_8ic_deinterleave_16i_x2)
    IfTypesThenDeinterleave(std::complex<int16_t>,int16_t,volk_16ic_deinterleave_16i_x2)
    IfTypesThenDeinterleave(std::complex<float>,float,volk_32fc_deinterleave_32f_x2)
    IfTypesThenDeinterleave(std::complex<float>,double,volk_32fc_deinterleave_64f_x2)

    throw InvalidDTypeException(
        VOLKDeinterleavePath,
        inDType,
        outDType);
}

static Pothos::BlockRegistry registerVOLKDeinterleave(
    VOLKDeinterleavePath,
    &makeDeinterleave);

/***********************************************************************
 * |PothosDoc Deinterleave Imag (VOLK)
 *
 * <p>
 * For each complex input, outputs the imaginary field.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32fc_deinterleave_imag_32f</b>
 * </p>
 *
 * |category /Convert
 * |category /Stream
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/deinterleave_imag()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKDeinterleaveImag(
    "/volk/deinterleave_imag",
    Pothos::Callable(OneToOneBlock<std::complex<float>,float>::make)
        .bind(volk_32fc_deinterleave_imag_32f, 0));

/***********************************************************************
 * |PothosDoc Deinterleave Real (VOLK)
 *
 * <p>
 * For each complex input, outputs the real field, performing type
 * conversions if needed.
 * </p>
 *
 * <p>
 * Supported types:
 * </p>
 *
 * <ul>
 *   <li>
 *     cint8 -> int8
 *     <ul>
 *       <li>Underlying function: <b>volk_8ic_deinterleave_real_8i</b></li>
 *     </ul>
 *   </li>
 *   <li>
 *     cint8 -> int16
 *     <ul>
 *       <li>Underlying function: <b>volk_8ic_deinterleave_real_16i</b></li>
 *       <li>Multiplies all output values by <b>256</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     cint16 -> int8
 *     <ul>
 *       <li>Underlying function: <b>volk_16ic_deinterleave_real_8i</b></li>
 *       <li>Divides all output values by <b>256</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     cint16 -> int16
 *     <ul>
 *       <li>Underlying function: <b>volk_16ic_deinterleave_real_16i</b></li>
 *     </ul>
 *   </li>
 *   <li>
 *     cfloat32 -> float32
 *     <ul>
 *       <li>Underlying function: <b>volk_32fc_deinterleave_real_32f</b></li>
 *     </ul>
 *   </li>
 *   <li>
 *     cfloat32 -> float64
 *     <ul>
 *       <li>Underlying function: <b>volk_32fc_deinterleave_real_64f</b></li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * |category /Convert
 * |category /Stream
 * |category /VOLK
 * |keywords complex
 *
 * |param inputDType[Data Type In]
 * |widget DTypeChooser(cint8=1,cint16=1,cfloat32=1)
 * |default "complex_float32"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(int8=1,int16=1,float32=1,float64=1)
 * |default "float32"
 * |preview disable
 *
 * |factory /volk/deinterleave_real(inputDType,outputDType)
 **********************************************************************/
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

    throw InvalidDTypeException(
        VOLKDeinterleaveRealPath,
        inDType,
        outDType);
}

static Pothos::BlockRegistry registerVOLKDeinterleaveReal(
    VOLKDeinterleaveRealPath,
    &makeDeinterleaveReal);

/***********************************************************************
 * |PothosDoc Deinterleave Real (Custom Scalar) (VOLK)
 *
 * <p>
 * For each complex input, outputs the real field, performing type
 * conversions and applying a given scalar. Whether the scalar is
 * multiplied or divided depends on the conversion and is listed below.
 * </p>
 *
 * <p>
 * Supported types:
 * </p>
 *
 * <ul>
 *   <li>
 *     cfloat32 -> int16 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_32fc_s32f_deinterleave_real_16i</b></li>
 *       <li>Multiplies all inputs by <b>scalar</b>.</li>
 *       <li>Truncates all scaled values to fit inside an <b>int16</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     cint8 -> float32 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_8ic_s32f_deinterleave_real_f32</b></li>
 *       <li>Divides all inputs by <b>scalar</b>.</li>
 *     </ul>
 *   </li>
 *   <li>
 *     cint16 -> float32 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_16ic_s32f_deinterleave_real_f32</b></li>
 *       <li>Divides all inputs by <b>scalar</b>.</li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * |category /Convert
 * |category /VOLK
 * |keywords type
 *
 * |param inputDType[Data Type In]
 * |widget DTypeChooser(cint8=1,cint16=1,cfloat32=1)
 * |default "complex_int16"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(int16=1,float32=1)
 * |default "float32"
 * |preview disable
 *
 * |param scalarDType[Scalar Data Type]
 * |widget DTypeChooser(float32=1)
 * |default "float32"
 * |preview disable
 *
 * |param scalar[Scalar] A scalar to apply to each input post-conversion.
 * |widget LineEdit()
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/deinterleave_real_scaled(inputDType,outputDType,scalarDType)
 * |setter setScalar(scalar)
 **********************************************************************/
static const std::string VOLKDeinterleaveRealScaledPath = "/volk/deinterleave_real_scaled";

static Pothos::Block* makeDeinterleaveRealScaled(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType,
    const Pothos::DType& scalarDType)
{
    IfTypesThenOneToOneScalarParamBlock(std::complex<int8_t>,float,float,"scalar","setScalar",volk_8ic_s32f_deinterleave_real_32f)
    IfTypesThenOneToOneScalarParamBlock(std::complex<int16_t>,float,float,"scalar","setScalar",volk_16ic_s32f_deinterleave_real_32f)
    IfTypesThenOneToOneScalarParamBlock(std::complex<float>,int16_t,float,"scalar","setScalar",volk_32fc_s32f_deinterleave_real_16i)

    throw InvalidDTypeException(
        VOLKDeinterleaveRealScaledPath,
        inDType,
        outDType,
        scalarDType);
}

static Pothos::BlockRegistry registerVOLKDeinterleaveRealScaled(
    VOLKDeinterleaveRealScaledPath,
    &makeDeinterleaveRealScaled);

/***********************************************************************
 * |PothosDoc Deinterleave (Custom Scalar) (VOLK)
 *
 * <p>
 * Deinterleaves a complex input into its real and imaginary inputs,
 * performing type conversions if needed. Multiplies each output by a
 * given scalar value.
 * </p>
 *
 * <p>
 * Supported types:
 * </p>
 *
 * <ul>
 *   <li>
 *     cint8 -> float32,float32 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_8ic_s32f_deinterleave_32f_x2</b></li>
 *     </ul>
 *   </li>
 *   <li>
 *     cint16 -> float32,float32 (float32 scalar)
 *     <ul>
 *       <li>Underlying function: <b>volk_16ic_s32f_deinterleave_32f_x2</b></li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * |category /Convert
 * |category /Stream
 * |category /VOLK
 * |keywords complex real imag
 *
 * |param inputDType[Data Type In]
 * |widget DTypeChooser(cint8=1,cint16=1)
 * |default "complex_int8"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(float32=1)
 * |default "float32"
 * |preview disable
 *
 * |param scalarDType[Scalar Data Type]
 * |widget DTypeChooser(float32=1)
 * |default "float32"
 * |preview disable
 *
 * |param scalar[Scalar] A scalar to apply to each input post-conversion.
 * |widget LineEdit()
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/deinterleave_scaled(inputDType,outputDType,scalarDType)
 * |setter setScalar(scalar)
 **********************************************************************/
static const std::string VOLKDeinterleaveScaledPath = "/volk/deinterleave_scaled";

static Pothos::Block* makeDeinterleaveScaled(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType,
    const Pothos::DType& scalarDType)
{
#define IfTypesThenDeinterleaveScaled(InType,OutType,Fcn) \
    IfTypesThenOneToTwoScalarParamBlock(InType,OutType,float,std::string,"scalar","setScalar",Fcn,"real","imag")

    IfTypesThenDeinterleaveScaled(std::complex<int8_t>,float,volk_8ic_s32f_deinterleave_32f_x2)
    IfTypesThenDeinterleaveScaled(std::complex<int16_t>,float,volk_16ic_s32f_deinterleave_32f_x2)

    throw InvalidDTypeException(
        VOLKDeinterleaveScaledPath,
        inDType,
        outDType,
        scalarDType);
}

static Pothos::BlockRegistry registerVOLKDeinterleaveScaled(
    VOLKDeinterleaveScaledPath,
    &makeDeinterleaveScaled);

/***********************************************************************
 * |PothosDoc Divide (VOLK)
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_32f_x2_divide_32f</b></li>
 * <li><b>volk_32fc_x2_divide_32fc</b></li>
 * </ul>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math
 *
 * |param input0DType[Data Type In0]
 * |widget DTypeChooser(float32=1,cfloat32=1)
 * |default "float32"
 * |preview disable
 *
 * |param input1DType[Data Type In1]
 * |widget DTypeChooser(float32=1,cfloat32=1)
 * |default "float32"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(float32=1,cfloat32=1)
 * |default "float32"
 * |preview disable
 *
 * |factory /volk/divide(input0DType,input1DType,outputDType)
 **********************************************************************/
static const std::string VOLKDividePath = "/volk/divide";

static Pothos::Block* makeDivide(
    const Pothos::DType& inDType0,
    const Pothos::DType& inDType1,
    const Pothos::DType& outDType)
{
#define IfTypeThenDivide(T,fcn) \
    IfTypesThenTwoToOneBlock(T,T,T,size_t,fcn,0,1)

    IfTypeThenDivide(float,volk_32f_x2_divide_32f)
    IfTypeThenDivide(std::complex<float>,volk_32fc_x2_divide_32fc)

    throw InvalidDTypeException(
        VOLKDividePath,
        std::vector<Pothos::DType>{inDType0, inDType1},
        outDType);
}

static Pothos::BlockRegistry registerVOLKDivide(
    VOLKDividePath,
    &makeDivide);

/***********************************************************************
 * |PothosDoc Exp (VOLK)
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 *   <li>Fast: <b>volk_32f_expfast_32f</b></li>
 *   <li>Precise: <b>volk_32f_exp_32f</b></li>
 * </ul>
 *
 * |param mode[Mode]
 * The <b>FAST</b> operation can have up to a <b>7%</b> error.
 * |widget ComboBox(editable=false)
 * |default "PRECISE"
 * |option [Fast] "FAST"
 * |option [Precise] "PRECISE"
 * |preview enable
 *
 * |category /Math
 * |category /VOLK
 * |keywords math
 *
 * |factory /volk/exp(mode)
 **********************************************************************/
static const std::string VOLKExpPath = "/volk/exp";

static Pothos::Block* makeExp(const std::string& mode)
{
    OneToOneFcn<float,float> volkFcn = nullptr;

    if(mode == "PRECISE")   volkFcn = volk_32f_exp_32f;
    else if(mode == "FAST") volkFcn = volk_32f_expfast_32f;
    else throw Pothos::InvalidArgumentException(VOLKExpPath + " mode: " + mode);

    assert(volkFcn);

    return OneToOneBlock<float,float>::make(volkFcn);
}

#warning TODO: why does this not show up in PothosFlow?
static Pothos::BlockRegistry registerVOLKExp(
    VOLKExpPath,
    &makeExp);

/***********************************************************************
 * |PothosDoc Interleave (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_x2_interleave_32fc</b>
 * </p>
 *
 * |category /Stream
 * |category /Convert
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/interleave()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKInterleave(
    "/volk/interleave",
    Pothos::Callable(TwoToOneBlock<float,float,std::complex<float>,std::string>::make)
        .bind(volk_32f_x2_interleave_32fc, 0)
        .bind("real", 1)
        .bind("imag", 2));

/***********************************************************************
 * |PothosDoc Interleave (Custom Scalar) (VOLK)
 *
 * <p>
 * Interleaves real and imaginary inputs into a complex output, then
 * applies a user-provided scalar.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32f_x2_s32f_interleave_16ic</b>
 * </p>
 *
 * |category /Convert
 * |category /Stream
 * |category /VOLK
 * |keywords complex real imag
 *
 * |param scalar[Scalar] A scalar to apply to each input post-conversion.
 * |widget DoubleSpinBox(decimals=3)
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/interleave_scaled()
 * |setter setScalar(scalar)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKInterleaveScaled(
    "/volk/interleave_scaled",
    Pothos::Callable(TwoToOneScalarParamBlock<float,float,std::complex<int16_t>,float,std::string>::make)
        .bind(volk_32f_x2_s32f_interleave_16ic, 0)
        .bind("scalar", 1)
        .bind("setScalar", 2)
        .bind("real", 3)
        .bind("imag", 4));

/***********************************************************************
 * |PothosDoc Inverse Square Root (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_invsqrt_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/invsqrt()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKInvSqrt(
    "/volk/invsqrt",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_invsqrt_32f, 0));

/***********************************************************************
 * |PothosDoc Log2 (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_log2_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/log2()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKLog2(
    "/volk/log2",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_log2_32f, 0));

/***********************************************************************
 * |PothosDoc Magnitude (VOLK)
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_16ic_magnitude_16i</b></li>
 * <li><b>volk_32fc_magnitude_32f</b></li>
 * </ul>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math complex
 *
 * |param inputDType[Data Type In]
 * |widget DTypeChooser(cint16=1,cfloat32=1)
 * |default "complex_float32"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(int16=1,float32=1)
 * |default "float32"
 * |preview disable
 *
 * |factory /volk/magnitude(inputDType,outputDType)
 **********************************************************************/
static const std::string VOLKMagnitudePath = "/volk/magnitude";

static Pothos::Block* makeMagnitude(const Pothos::DType& inDType, const Pothos::DType& outDType)
{
    IfTypesThenOneToOneBlock(std::complex<int16_t>,int16_t,volk_16ic_magnitude_16i)
    IfTypesThenOneToOneBlock(std::complex<float>,float,volk_32fc_magnitude_32f)

    throw InvalidDTypeException(VOLKMagnitudePath, inDType, outDType);
}

static Pothos::BlockRegistry registerVOLKMagnitude(
    VOLKMagnitudePath,
    &makeMagnitude);

/***********************************************************************
 * |PothosDoc Magnitude Squared (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_magnitude_squared_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math complex
 *
 * |factory /volk/magnitude_squared()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKMagnitudeSquared(
    "/volk/magnitude_squared",
    Pothos::Callable(OneToOneBlock<std::complex<float>,float>::make)
        .bind(volk_32fc_magnitude_squared_32f, 0));

/***********************************************************************
 * |PothosDoc Max (VOLK)
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_32f_x2_max_32f</b></li>
 * <li><b>volk_64f_x2_max_64f</b></li>
 * </ul>
 *
 * |category /Stream
 * |category /VOLK
 *
 * |param dtype[Data Type]
 * |widget DTypeChooser(float=1)
 * |default "float32"
 * |preview disable
 *
 * |factory /volk/max(dtype)
 **********************************************************************/
static const std::string VOLKMaxPath = "/volk/max";

static Pothos::Block* makeMax(const Pothos::DType& dtype)
{
#define IfTypeThenMax(T,fcn) \
    IfTypeThenTwoToOneBlock(T,size_t,fcn,0,1)

    IfTypeThenMax(float,volk_32f_x2_max_32f)
    IfTypeThenMax(double,volk_64f_x2_max_64f)

    throw InvalidDTypeException(VOLKMaxPath, dtype);
}

static Pothos::BlockRegistry registerVOLKMax(
    VOLKMaxPath,
    &makeMax);

/***********************************************************************
 * |PothosDoc Max* (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_16i_max_star_horizontal_16i</b>
 * </p>
 *
 * |category /VOLK
 *
 * |factory /volk/max_star()
 **********************************************************************/
static const std::string VOLKMaxStarPath = "/volk/max_star_horizontal_path";

// For some reason, the signature is different for this function,
// so we need this lambda for it to match.
static const auto VOLKMaxStar = [](int16_t* out, const int16_t* in, unsigned int len)
{
    return volk_16i_max_star_horizontal_16i(out, const_cast<int16_t*>(in), len);
};

static Pothos::BlockRegistry registerVOLKMaxStarPath(
    "/volk/max_star",
    Pothos::Callable(OneToOneBlock<int16_t,int16_t>::make)
        .bind<OneToOneFcn<int16_t,int16_t>>(VOLKMaxStar, 0));

/***********************************************************************
 * |PothosDoc Min (VOLK)
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_32f_x2_min_32f</b></li>
 * <li><b>volk_64f_x2_min_64f</b></li>
 * </ul>
 *
 * |category /Stream
 * |category /VOLK
 *
 * |param dtype[Data Type]
 * |widget DTypeChooser(float=1)
 * |default "float32"
 * |preview disable
 *
 * |factory /volk/min(dtype)
 **********************************************************************/
static const std::string VOLKMinPath = "/volk/min";

static Pothos::Block* makeMin(const Pothos::DType& dtype)
{
#define IfTypeThenMin(T,fcn) \
    IfTypeThenTwoToOneBlock(T,size_t,fcn,0,1)

    IfTypeThenMin(float,volk_32f_x2_min_32f)
    IfTypeThenMin(double,volk_64f_x2_min_64f)

    throw InvalidDTypeException(VOLKMinPath, dtype);
}

static Pothos::BlockRegistry registerVOLKMin(
    VOLKMinPath,
    &makeMin);

/***********************************************************************
 * |PothosDoc Multiply (VOLK)
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_32f_64f_multiply_64f</b></li>
 * <li><b>volk_64f_x2_multiply_64f</b></li>
 * <li><b>volk_16ic_x2_multiply_16ic</b></li>
 * <li><b>volk_32fc_x2_multiply_32fc</b></li>
 * <li><b>volk_32fc_32f_multiply_32fc</b></li>
 * </ul>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math plus
 *
 * |param input0DType[Data Type In0]
 * |widget DTypeChooser(float=1,cint16=1,cfloat32=1)
 * |default "float64"
 * |preview disable
 *
 * |param input1DType[Data Type In1]
 * |widget DTypeChooser(float=1,cint16=1,cfloat32=1)
 * |default "float64"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(float64=1,cint16=1,cfloat32=1)
 * |default "float64"
 * |preview disable
 *
 * |factory /volk/multiply(input0DType,input1DType,outputDType)
 **********************************************************************/
static const std::string VOLKMultiplyPath = "/volk/multiply";

static Pothos::Block* makeMultiply(
    const Pothos::DType& inDType0,
    const Pothos::DType& inDType1,
    const Pothos::DType& outDType)
{
#define IfTypesThenMultiply(InType0,InType1,OutType,fcn) \
    IfTypesThenTwoToOneBlock(InType0,InType1,OutType,size_t,fcn,0,1)

    IfTypesThenMultiply(float,double,double,volk_32f_64f_multiply_64f)
    IfTypesThenMultiply(double,double,double,volk_64f_x2_multiply_64f)
    IfTypesThenMultiply(std::complex<int16_t>,std::complex<int16_t>,std::complex<int16_t>,volk_16ic_x2_multiply_16ic)
    IfTypesThenMultiply(std::complex<float>,std::complex<float>,std::complex<float>,volk_32fc_x2_multiply_32fc)
    IfTypesThenMultiply(std::complex<float>,float,std::complex<float>,volk_32fc_32f_multiply_32fc)

    throw InvalidDTypeException(
        VOLKMultiplyPath,
        std::vector<Pothos::DType>{inDType0, inDType1},
        outDType);
}

static Pothos::BlockRegistry registerVOLKMultiply(
    VOLKMultiplyPath,
    &makeMultiply);

/***********************************************************************
 * |PothosDoc Multiply Conjugate (VOLK)
 *
 * <p>
 * For each input pair, multiplies the first number by the complex
 * conjugate of the second number.
 * </p>
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_8ic_x2_multiply_conjugate_16ic</b></li>
 * <li><b>volk_32fc_x2_multiply_conjugate_32fc</b></li>
 * </ul>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math complex
 *
 * |param input0DType[Data Type In0]
 * |widget DTypeChooser(cint8=1,cfloat32=1)
 * |default "complex_float32"
 * |preview disable
 *
 * |param input1DType[Data Type In1]
 * |widget DTypeChooser(cint8=1,cfloat32=1)
 * |default "complex_float32"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(cint16=1,cfloat32=1)
 * |default "complex_float32"
 * |preview disable
 *
 * |factory /volk/multiply_conjugate(input0DType,input1DType,outputDType)
 **********************************************************************/
static const std::string VOLKMultiplyConjugatePath = "/volk/multiply_conjugate";

static Pothos::Block* makeMultiplyConjugate(
    const Pothos::DType& inDType0,
    const Pothos::DType& inDType1,
    const Pothos::DType& outDType)
{
#define IfTypesThenMultiplyConjugate(InType0,InType1,OutType,fcn) \
    IfTypesThenTwoToOneBlock(InType0,InType1,OutType,size_t,fcn,0,1)

    IfTypesThenMultiplyConjugate(std::complex<int8_t>,std::complex<int8_t>,std::complex<int16_t>,volk_8ic_x2_multiply_conjugate_16ic)
    IfTypesThenMultiplyConjugate(std::complex<float>,std::complex<float>,std::complex<float>,volk_32fc_x2_multiply_conjugate_32fc)

    throw InvalidDTypeException(
        VOLKMultiplyConjugatePath,
        std::vector<Pothos::DType>{inDType0, inDType1},
        outDType);
}

static Pothos::BlockRegistry registerVOLKMultiplyConjugate(
    VOLKMultiplyConjugatePath,
    &makeMultiplyConjugate);

/***********************************************************************
 * |PothosDoc Multiply Conjugate Add (VOLK)
 *
 * <p>
 * Add each element of the first vector to the complex conjugate of its
 * corresponding value in the second vector (multiplied by a given constant).
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32fc_x2_s32fc_multiply_conjugate_add_32fc</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords complex real imag
 *
 * |param scalar[Complex Scalar]
 * A complex value to apply to each input of the second vector.
 * |widget LineEdit()
 * |default 1.0+0i
 * |preview enable
 *
 * |factory /volk/multiply_conjugate_add()
 * |setter setScalar(scalar)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKMultiplyConjugateAdd(
    "/volk/multiply_conjugate_add",
    Pothos::Callable(TwoToOneScalarParamBlock<std::complex<float>,std::complex<float>,std::complex<float>,std::complex<float>,size_t>::make)
        .bind(volk_32fc_x2_s32fc_multiply_conjugate_add_32fc, 0)
        .bind("scalar", 1)
        .bind("setScalar", 2)
        .bind(0, 3)
        .bind(1, 4));

/***********************************************************************
 * |PothosDoc Multiply Conjugate (Custom Scalar) (VOLK)
 *
 * <p>
 * For each input pair, multiplies the first number by the scaled complex
 * conjugate of the second number.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_8ic_x2_s32f_multiply_conjugate_32fc</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math complex
 *
 * |param scalar[Scalar]
 * A constant value multiplied with all values in the second vector.
 * |widget DoubleSpinBox(decimals=3)
 * |default 0.0
 * |preview enable
 *
 * |factory /volk/multiply_conjugate_scaled()
 * |setter setScalar(scalar)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKMultiplyConjugateScaled(
    "/volk/multiply_conjugate_scaled",
    Pothos::Callable(TwoToOneScalarParamBlock<std::complex<int8_t>,std::complex<int8_t>,std::complex<float>,float,size_t>::make)
        .bind(volk_8ic_x2_s32f_multiply_conjugate_32fc, 0)
        .bind("scalar", 1)
        .bind("setScalar", 2)
        .bind(0, 3)
        .bind(1, 4));

/***********************************************************************
 * |PothosDoc Scalar Multiply (VOLK)
 *
 * <p>
 * Multiplies a given scalar constant to all elements.
 * </p>
 *
 * <p>
 * Underlying functions:
 * </p>
 *
 * <ul>
 * <li><b>volk_32f_s32f_multiply_32f</b></li>
 * <li><b>volk_32fc_s32fc_multiply_32fc</b></li>
 * </ul>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math constant
 *
 * |param inputDType[Data Type In]
 * |widget DTypeChooser(float32=1,cfloat32=1)
 * |default "float32"
 * |preview disable
 *
 * |param outputDType[Data Type Out]
 * |widget DTypeChooser(float32=1,cfloat32=1)
 * |default "float32"
 * |preview disable
 *
 * |param scalarDType[Scalar Data Type]
 * |widget DTypeChooser(float32=1,cfloat32=1)
 * |default "float32"
 * |preview disable
 *
 * |param scalar[Scalar] A constant value multiplied with all inputs.
 * |widget LineEdit()
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/multiply_scalar(inputDType,outputDType,scalarDType)
 * |setter setScalar(scalar)
 **********************************************************************/
static const std::string VOLKMultiplyScalarPath = "/volk/multiply_scalar";

static Pothos::Block* makeMultiplyScalar(
    const Pothos::DType& inDType,
    const Pothos::DType& outDType,
    const Pothos::DType& scalarDType)
{
#define IfTypesThenMultiplyScalar(InType,OutType,ScalarType,Fcn) \
    IfTypesThenOneToOneScalarParamBlock(InType,OutType,ScalarType,"scalar","setScalar",Fcn)

    IfTypesThenMultiplyScalar(float,float,float,volk_32f_s32f_multiply_32f)
    IfTypesThenMultiplyScalar(std::complex<float>,std::complex<float>,std::complex<float>,volk_32fc_s32fc_multiply_32fc)

    throw InvalidDTypeException(
        VOLKConvertPath,
        std::vector<Pothos::DType>{inDType, outDType},
        scalarDType);
}

static Pothos::BlockRegistry registerVOLKMultiplyScalar(
    VOLKMultiplyScalarPath,
    &makeMultiplyScalar);

/***********************************************************************
 * |PothosDoc Logical Or (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32i_x2_or_32i</b>
 * </p>
 *
 * |category /Digital
 * |category /VOLK
 *
 * |factory /volk/or()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKOr(
    "/volk/or",
    Pothos::Callable(TwoToOneBlock<int,int,int,size_t>::make)
        .bind(volk_32i_x2_or_32i, 0)
        .bind(0, 1)
        .bind(1, 2));

/***********************************************************************
 * |PothosDoc Pow (VOLK)
 *
 * <p>
 * Raises each element in the <b>"input"</b> port by its corresponding
 * element in the <b>"exp"</b> port.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32f_x2_pow_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 *
 * |factory /volk/pow()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKPow(
    "/volk/pow",
    Pothos::Callable(TwoToOneBlock<float,float,float,std::string>::make)
        .bind(volk_32f_x2_pow_32f, 0)
        .bind("exp", 1)
        .bind("input", 2));

/***********************************************************************
 * |PothosDoc Scalar Pow (VOLK)
 *
 * <p>
 * Raises each element to a user-given power.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32f_s32f_power_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords exponent
 *
 * |param power[Power] A scalar exponent to apply to all elements.
 * |widget DoubleSpinBox(decimals=3)
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/power()
 * |setter setPower(power)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKPower(
    "/volk/power",
    Pothos::Callable(OneToOneScalarParamBlock<float,float,float>::make)
        .bind(volk_32f_s32f_power_32f, 0)
        .bind("power", 1)
        .bind("setPower", 2));

/***********************************************************************
 * |PothosDoc Power Spectrum (VOLK)
 *
 * <p>
 * Calculates the log10 power value for each input.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32fc_s32f_power_spectrum_32f</b>
 * </p>
 *
 * |category /Math
 * |category /FFT
 * |category /VOLK
 * |keywords math rf
 *
 * |param normalizationFactor[Normalization Factor]
 * A value multiplied to all outputs.
 * |widget DoubleSpinBox(decimals=3)
 * |default 1.0
 * |preview enable
 *
 * |factory /volk/power_spectrum()
 * |setter setNormalizationFactor(normalizationFactor)
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKPowerSpectrum(
    "/volk/power_spectrum",
    Pothos::Callable(OneToOneScalarParamBlock<std::complex<float>,float,float>::make)
        .bind(volk_32fc_s32f_power_spectrum_32f, 0)
        .bind("normalizationFactor", 1)
        .bind("setNormalizationFactor", 2));

/***********************************************************************
 * |PothosDoc Bitwise Reverse (VOLK)
 *
 * <p>
 * Reverses the bits in each input.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32u_reverse_32u</b>
 * </p>
 *
 * |category /Digital
 * |category /VOLK
 *
 * |factory /volk/reverse()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKReverse(
    "/volk/reverse",
    Pothos::Callable(OneToOneBlock<uint32_t,uint32_t>::make)
        .bind(volk_32u_reverse_32u, 0));

/***********************************************************************
 * |PothosDoc Sin (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_sin_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/sin()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKSin(
    "/volk/sin",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_sin_32f, 0));

/***********************************************************************
 * |PothosDoc Square Root (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_sqrt_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/sqrt()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKSqrt(
    "/volk/sqrt",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_sqrt_32f, 0));

/***********************************************************************
 * |PothosDoc Subtract (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_x2_subtract_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math minus
 *
 * |factory /volk/subtract()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKSubtract(
    "/volk/subtract",
    Pothos::Callable(TwoToOneBlock<float,float,float,size_t>::make)
        .bind(volk_32f_x2_subtract_32f, 0)
        .bind(0, 1)
        .bind(1, 2));

/***********************************************************************
 * |PothosDoc Tan (VOLK)
 *
 * <p>
 * Underlying function: <b>volk_32f_tan_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/tan()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKTan(
    "/volk/tan",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_tan_32f, 0));

/***********************************************************************
 * |PothosDoc TanH (VOLK)
 *
 * <p>
 * Calculates the hyperbolic tangent of each input.
 * </p>
 *
 * <p>
 * Underlying function: <b>volk_32f_tanh_32f</b>
 * </p>
 *
 * |category /Math
 * |category /VOLK
 * |keywords math trig
 *
 * |factory /volk/tanh()
 **********************************************************************/
static Pothos::BlockRegistry registerVOLKTanH(
    "/volk/tanh",
    Pothos::Callable(OneToOneBlock<float,float>::make)
        .bind(volk_32f_tanh_32f, 0));
