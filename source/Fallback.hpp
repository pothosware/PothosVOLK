// Copyright 2021 Nicholas Corgan
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
