// Copyright 2021,2023 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <volk/volk.h>

// In order to support versions of VOLK used in some package
// managers as of the initial development of this code, the
// generic implementations of the functions will be in this
// header to be used if the installed version of VOLK is
// earlier than when the function was added.

#ifndef HAVE_32FC_ACCUMULATOR
static inline void volk_32fc_accumulator_s32fc(lv_32fc_t* result,
                                               const lv_32fc_t* inputBuffer,
                                               unsigned int num_points)
{
    const lv_32fc_t* aPtr = inputBuffer;
    unsigned int number = 0;
    lv_32fc_t returnValue = lv_cmake(0.f, 0.f);

    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif

#ifndef HAVE_32F_S32F_ADD
static inline void volk_32f_s32f_add_32f(float* cVector,
                                         const float* aVector,
                                         const float scalar,
                                         unsigned int num_points)
{
    unsigned int number = 0;
    const float* inputPtr = aVector;
    float* outputPtr = cVector;
    for (number = 0; number < num_points; number++) {
        *outputPtr = (*inputPtr) + scalar;
        inputPtr++;
        outputPtr++;
    }
}
#endif

#ifndef HAVE_32F_EXP
static inline void
volk_32f_exp_32f(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *bPtr++ = expf(*aPtr++);
    }
}
#endif

#ifndef HAVE_32FC_X2_S32FC_MULTIPLY_CONJUGATE_ADD
static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc(lv_32fc_t* cVector,
                                               const lv_32fc_t* aVector,
                                               const lv_32fc_t* bVector,
                                               const lv_32fc_t scalar,
                                               unsigned int num_points)
{
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;
    lv_32fc_t* cPtr = cVector;
    unsigned int number = num_points;

    // unwrap loop
    while (number >= 8) {
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        number -= 8;
    }

    // clean up any remaining
    while (number-- > 0) {
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
    }
}
#endif

// In order to support versions of VOLK used later than the
// initial development of this code, the generic
// implementations of deprecated functions will be in this
// header to be used if the installed version of VOLK is
// later than when the function was removed.

#ifndef HAVE_16I_MAX_STAR
static inline void
volk_16i_max_star_16i(short* target, short* src0, unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    short candidate = src0[0];
    for (i = 1; i < bound; ++i) {
        candidate = ((short)(candidate - src0[i]) > 0) ? candidate : src0[i];
    }
    target[0] = candidate;
#endif

#ifndef HAVE_16I_X4_QUAD_MAX_STAR
static inline void volk_16i_x4_quad_max_star_16i(short* target,
                                                 short* src0,
                                                 short* src1,
                                                 short* src2,
                                                 short* src3,
                                                 unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    short temp0 = 0;
    short temp1 = 0;
    for (i = 0; i < bound; ++i) {
        temp0 = ((short)(src0[i] - src1[i]) > 0) ? src0[i] : src1[i];
        temp1 = ((short)(src2[i] - src3[i]) > 0) ? src2[i] : src3[i];
        target[i] = ((short)(temp0 - temp1) > 0) ? temp0 : temp1;
    }
}
#endif

#ifndef HAVE_16I_X5_ADD_QUAD
static inline void volk_16i_x5_add_quad_16i_x4(short* target0,
                                               short* target1,
                                               short* target2,
                                               short* target3,
                                               short* src0,
                                               short* src1,
                                               short* src2,
                                               short* src3,
                                               short* src4,
                                               unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    for (i = 0; i < bound; ++i) {
        target0[i] = src0[i] + src1[i];
        target1[i] = src0[i] + src2[i];
        target2[i] = src0[i] + src3[i];
        target3[i] = src0[i] + src4[i];
    }
}
#endif
