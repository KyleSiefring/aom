/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef DAALA_TX_AVX2_H_
#define DAALA_TX_AVX2_H_

#include <tmmintrin.h>
#include <immintrin.h>

#if CONFIG_DAALA_TX

static INLINE __m128i od_unbiased_rshift1_epi16(__m128i a) {
  return _mm_srai_epi16(_mm_add_epi16(_mm_srli_epi16(a, 15), a), 1);
}

static INLINE __m256i od_mm256_unbiased_rshift1_epi16(__m256i a) {
  return _mm256_srai_epi16(_mm256_add_epi16(_mm256_srli_epi16(a, 15), a), 1);
}

static INLINE __m256i od_mm256_unbiased_rshift1_epi32(__m256i a) {
  return _mm256_srai_epi32(_mm256_add_epi32(_mm256_srli_epi32(a, 31), a), 1);
}

static INLINE __m128i od_avg_epi16(__m128i a, __m128i b) {
  __m128i sign_bit;
  /*x86 only provides an unsigned PAVGW with a bias (ARM is better here).
    We emulate a signed one by adding an offset to convert to unsigned and
    back. We use XOR instead of addition/subtraction because it dispatches
    better on older processors.*/
  sign_bit = _mm_set1_epi16(0x8000);
  return _mm_xor_si128(
      _mm_avg_epu16(_mm_xor_si128(a, sign_bit), _mm_xor_si128(b, sign_bit)),
      sign_bit);
}

static INLINE __m256i od_mm256_avg_epi16(__m256i a, __m256i b) {
  __m256i sign_bit;
  sign_bit = _mm256_set1_epi16(0x8000);
  return _mm256_xor_si256(_mm256_avg_epu16(_mm256_xor_si256(a, sign_bit),
                                           _mm256_xor_si256(b, sign_bit)),
                          sign_bit);
}

static INLINE __m256i od_mm256_avg_epi32(__m256i a, __m256i b) {
  __m256i neg1;
  /* It's cheaper to generate -1's than 1's. */
  neg1 = _mm256_set1_epi64x(-1);
  /* There is no corresponding PAVGD, but we are not in danger of overflowing
     a 32-bit register. */
  return _mm256_srai_epi32(_mm256_add_epi32(a, _mm256_sub_epi32(b, neg1)), 1);
}

/*Like the above, but does (a - b + 1) >> 1 instead.*/
static INLINE __m128i od_hrsub_epi16(__m128i a, __m128i b) {
  __m128i sign_bit;
  sign_bit = _mm_set1_epi16(0x8000);
  return _mm_xor_si128(
      _mm_avg_epu16(_mm_xor_si128(a, sign_bit), _mm_sub_epi16(sign_bit, b)),
      sign_bit);
}

static INLINE __m256i od_mm256_hrsub_epi16(__m256i a, __m256i b) {
  __m256i sign_bit;
  sign_bit = _mm256_set1_epi16(0x8000);
  return _mm256_xor_si256(_mm256_avg_epu16(_mm256_xor_si256(a, sign_bit),
                                           _mm256_sub_epi16(sign_bit, b)),
                          sign_bit);
}

static INLINE __m256i od_mm256_hrsub_epi32(__m256i a, __m256i b) {
  __m256i neg1;
  /* It's cheaper to generate -1's than 1's. */
  neg1 = _mm256_set1_epi64x(-1);
  /* There is no corresponding PAVGD, but we are not in danger of overflowing
     a 32-bit register. */
  return _mm256_srai_epi32(_mm256_sub_epi32(a, _mm256_add_epi32(b, neg1)), 1);
}

static INLINE void od_swap_si128(__m128i *q0, __m128i *q1) {
  __m128i t;
  t = *q0;
  *q0 = *q1;
  *q1 = t;
}

static INLINE void od_mm256_swap_si256(__m256i *q0, __m256i *q1) {
  __m256i t;
  t = *q0;
  *q0 = *q1;
  *q1 = t;
}

static INLINE __m128i od_mulhrs_epi16(__m128i a, int16_t b) {
  return _mm_mulhrs_epi16(a, _mm_set1_epi16(b));
}

static INLINE __m128i od_mul_epi16(__m128i a, int32_t b, int r) {
  int32_t b_q15;
  b_q15 = b << (15 - r);
  /* b and r are in all cases compile-time constants, so these branches
     disappear when this function gets inlined. */
  if (b_q15 > 32767) {
    return _mm_add_epi16(a, od_mulhrs_epi16(a, (int16_t)(b_q15 - 32768)));
  } else if (b_q15 < -32767) {
    return _mm_sub_epi16(od_mulhrs_epi16(a, (int16_t)(32768 + b_q15)), a);
  } else {
    return od_mulhrs_epi16(a, b_q15);
  }
}

static INLINE __m256i od_mm256_mulhrs_epi16(__m256i a, int16_t b) {
  return _mm256_mulhrs_epi16(a, _mm256_set1_epi16(b));
}

static INLINE __m256i od_mm256_mul_epi16(__m256i a, int32_t b, int r) {
  int32_t b_q15;
  b_q15 = b << (15 - r);
  /* b and r are in all cases compile-time constants, so these branches
     disappear when this function gets inlined. */
  if (b_q15 > 32767) {
    return _mm256_add_epi16(a,
                            od_mm256_mulhrs_epi16(a, (int16_t)(b_q15 - 32768)));
  } else if (b_q15 < -32767) {
    return _mm256_sub_epi16(od_mm256_mulhrs_epi16(a, (int16_t)(32768 + b_q15)),
                            a);
  } else {
    return od_mm256_mulhrs_epi16(a, b_q15);
  }
}

static INLINE __m256i od_mm256_mul_epi32(__m256i a, int32_t b, int r) {
  __m256i neg1;
  /* It's cheaper to generate -1's than 1's. */
  neg1 = _mm256_set1_epi64x(-1);
  /* There's no 32-bit version of PMULHRSW on x86 like there is on ARM .*/
  a = _mm256_mullo_epi32(a, _mm256_set1_epi32(b));
  a = _mm256_srai_epi32(a, r - 1);
  a = _mm256_sub_epi32(a, neg1);
  return _mm256_srai_epi32(a, 1);
}

static INLINE __m128i od_hbd_max_epi16(int bd) {
  return _mm_set1_epi16((1 << bd) - 1);
}

static INLINE __m256i od_mm256_hbd_max_epi16(int bd) {
  return _mm256_set1_epi16((1 << bd) - 1);
}

static INLINE __m128i od_hbd_clamp_epi16(__m128i a, __m128i max) {
  return _mm_max_epi16(_mm_setzero_si128(), _mm_min_epi16(a, max));
}

static INLINE __m256i od_mm256_hbd_clamp_epi16(__m256i a, __m256i max) {
  return _mm256_max_epi16(_mm256_setzero_si256(), _mm256_min_epi16(a, max));
}

/* Loads a 4x4 buffer of 32-bit values into four SSE registers. */
static INLINE void od_load_buffer_4x4_epi32(__m128i *q0, __m128i *q1,
                                            __m128i *q2, __m128i *q3,
                                            const tran_low_t *in) {
  *q0 = _mm_loadu_si128((const __m128i *)in + 0);
  *q1 = _mm_loadu_si128((const __m128i *)in + 1);
  *q2 = _mm_loadu_si128((const __m128i *)in + 2);
  *q3 = _mm_loadu_si128((const __m128i *)in + 3);
}

/* Loads a 4x4 buffer of 16-bit values into four SSE registers. */
static INLINE void od_load_buffer_4x4_epi16(__m128i *q0, __m128i *q1,
                                            __m128i *q2, __m128i *q3,
                                            const int16_t *in) {
  *q0 = _mm_loadu_si128((const __m128i *)in + 0);
  *q1 = _mm_unpackhi_epi64(*q0, *q0);
  *q2 = _mm_loadu_si128((const __m128i *)in + 1);
  *q3 = _mm_unpackhi_epi64(*q2, *q2);
}

/* Loads an 8x4 buffer of 16-bit values into four SSE registers. */
static INLINE void od_load_buffer_8x4_epi16(__m128i *q0, __m128i *q1,
                                            __m128i *q2, __m128i *q3,
                                            const int16_t *in, int in_stride) {
  *q0 = _mm_loadu_si128((const __m128i *)(in + 0 * in_stride));
  *q1 = _mm_loadu_si128((const __m128i *)(in + 1 * in_stride));
  *q2 = _mm_loadu_si128((const __m128i *)(in + 2 * in_stride));
  *q3 = _mm_loadu_si128((const __m128i *)(in + 3 * in_stride));
}

/* Loads an 8x4 buffer of 32-bit values and packs them into 16-bit values in
   four SSE registers. */
static INLINE void od_load_pack_buffer_8x4_epi32(__m128i *r0, __m128i *r1,
                                                 __m128i *r2, __m128i *r3,
                                                 const tran_low_t *in) {
  __m128i r4;
  __m128i r5;
  __m128i r6;
  __m128i r7;
  *r0 = _mm_loadu_si128((const __m128i *)in + 0);
  r4 = _mm_loadu_si128((const __m128i *)in + 1);
  *r1 = _mm_loadu_si128((const __m128i *)in + 2);
  r5 = _mm_loadu_si128((const __m128i *)in + 3);
  *r2 = _mm_loadu_si128((const __m128i *)in + 4);
  r6 = _mm_loadu_si128((const __m128i *)in + 5);
  *r3 = _mm_loadu_si128((const __m128i *)in + 6);
  r7 = _mm_loadu_si128((const __m128i *)in + 7);
  *r0 = _mm_packs_epi32(*r0, r4);
  *r1 = _mm_packs_epi32(*r1, r5);
  *r2 = _mm_packs_epi32(*r2, r6);
  *r3 = _mm_packs_epi32(*r3, r7);
}

/* Loads an 8x4 buffer of 32-bit values into four AVX registers. */
static INLINE void od_load_buffer_8x4_epi32(__m256i *r0, __m256i *r1,
                                            __m256i *r2, __m256i *r3,
                                            const tran_low_t *in) {
  *r0 = _mm256_loadu_si256((const __m256i *)in + 0);
  *r1 = _mm256_loadu_si256((const __m256i *)in + 1);
  *r2 = _mm256_loadu_si256((const __m256i *)in + 2);
  *r3 = _mm256_loadu_si256((const __m256i *)in + 3);
}

/* Loads a 16x4 buffer of 16-bit values into four AVX registers. */
static INLINE void od_load_buffer_16x4_epi16(__m256i *r0, __m256i *r1,
                                             __m256i *r2, __m256i *r3,
                                             const int16_t *in, int in_stride) {
  *r0 = _mm256_loadu_si256((const __m256i *)(in + 0 * in_stride));
  *r1 = _mm256_loadu_si256((const __m256i *)(in + 1 * in_stride));
  *r2 = _mm256_loadu_si256((const __m256i *)(in + 2 * in_stride));
  *r3 = _mm256_loadu_si256((const __m256i *)(in + 3 * in_stride));
}

/* Loads a 4x4 buffer of 16-bit values, shifts them up to full bit depth, and
   puts them into four SSE registers. */
static INLINE void od_load_buffer_hbd_4x4_epi16(__m128i *q0, __m128i *q1,
                                                __m128i *q2, __m128i *q3,
                                                const int16_t *in,
                                                int input_stride, int bd) {
  const int upshift = TX_COEFF_DEPTH - bd;
  *q0 = _mm_loadl_epi64((const __m128i *)(in + 0 * input_stride));
  *q1 = _mm_loadl_epi64((const __m128i *)(in + 1 * input_stride));
  *q2 = _mm_loadl_epi64((const __m128i *)(in + 2 * input_stride));
  *q3 = _mm_loadl_epi64((const __m128i *)(in + 3 * input_stride));
  *q0 = _mm_slli_epi16(*q0, upshift);
  *q1 = _mm_slli_epi16(*q1, upshift);
  *q2 = _mm_slli_epi16(*q2, upshift);
  *q3 = _mm_slli_epi16(*q3, upshift);
}

/* Loads a 8x4 buffer of 16-bit values, shifts them up to full bit depth, and
   puts them into four SSE registers. */
static INLINE void od_load_buffer_hbd_8x4_epi16(__m128i *q0, __m128i *q1,
                                                __m128i *q2, __m128i *q3,
                                                const int16_t *in,
                                                int input_stride, int bd) {
  const int upshift = TX_COEFF_DEPTH - bd;
  *q0 = _mm_loadu_si128((const __m128i *)(in + 0 * input_stride));
  *q1 = _mm_loadu_si128((const __m128i *)(in + 1 * input_stride));
  *q2 = _mm_loadu_si128((const __m128i *)(in + 2 * input_stride));
  *q3 = _mm_loadu_si128((const __m128i *)(in + 3 * input_stride));
  *q0 = _mm_slli_epi16(*q0, upshift);
  *q1 = _mm_slli_epi16(*q1, upshift);
  *q2 = _mm_slli_epi16(*q2, upshift);
  *q3 = _mm_slli_epi16(*q3, upshift);
}

/* Stores a 4x4 buffer of 16-bit values from two SSE registers.
   Each register holds two rows of values. */
static INLINE void od_store_buffer_4x4_epi16(int16_t *out, __m128i q0,
                                             __m128i q1) {
  _mm_storeu_si128((__m128i *)out + 0, q0);
  _mm_storeu_si128((__m128i *)out + 1, q1);
}

/* Stores a 4x8 buffer of 16-bit values from four SSE registers.
   Each register holds two rows of values. */
static INLINE void od_store_buffer_4x8_epi16(int16_t *out, __m128i q0,
                                             __m128i q1, __m128i q2,
                                             __m128i q3) {
  _mm_storeu_si128((__m128i *)out + 0, q0);
  _mm_storeu_si128((__m128i *)out + 1, q1);
  _mm_storeu_si128((__m128i *)out + 2, q2);
  _mm_storeu_si128((__m128i *)out + 3, q3);
}

static INLINE void od_store_buffer_2x16_epi16(int16_t *out, __m256i r0,
                                              __m256i r1) {
  _mm256_storeu_si256((__m256i *)out + 0, r0);
  _mm256_storeu_si256((__m256i *)out + 1, r1);
}

/* Stores a 4x4 buffer of 32-bit values from four SSE registers. */
static INLINE void od_store_buffer_4x4_epi32(tran_low_t *out, __m128i q0,
                                             __m128i q1, __m128i q2,
                                             __m128i q3) {
  _mm_storeu_si128((__m128i *)out + 0, q0);
  _mm_storeu_si128((__m128i *)out + 1, q1);
  _mm_storeu_si128((__m128i *)out + 2, q2);
  _mm_storeu_si128((__m128i *)out + 3, q3);
}

/* Stores a 8x4 buffer of 32-bit values from eight SSE registers. */
static INLINE void od_store_buffer_8x4_epi32(tran_low_t *out, __m128i q0,
                                             __m128i q1, __m128i q2, __m128i q3,
                                             __m128i q4, __m128i q5, __m128i q6,
                                             __m128i q7) {
  _mm_storeu_si128((__m128i *)out + 0, q0);
  _mm_storeu_si128((__m128i *)out + 1, q1);
  _mm_storeu_si128((__m128i *)out + 2, q2);
  _mm_storeu_si128((__m128i *)out + 3, q3);
  _mm_storeu_si128((__m128i *)out + 4, q4);
  _mm_storeu_si128((__m128i *)out + 5, q5);
  _mm_storeu_si128((__m128i *)out + 6, q6);
  _mm_storeu_si128((__m128i *)out + 7, q7);
}

/* Loads a 4x4 buffer of 16-bit values, adds a 4x4 block of 16-bit values to
   them, clamps to high bit depth, and stores the sum back. */
static INLINE void od_add_store_buffer_hbd_4x4_epi16(void *output_pixels,
                                                     int output_stride,
                                                     __m128i q0, __m128i q1,
                                                     __m128i q2, __m128i q3,
                                                     int bd) {
  uint16_t *output_pixels16;
  __m128i p0;
  __m128i p1;
  __m128i p2;
  __m128i p3;
  __m128i max;
  __m128i round;
  int downshift;
  output_pixels16 = CONVERT_TO_SHORTPTR(output_pixels);
  max = od_hbd_max_epi16(bd);
  downshift = TX_COEFF_DEPTH - bd;
  round = _mm_set1_epi16((1 << downshift) >> 1);
  p0 = _mm_loadl_epi64((const __m128i *)(output_pixels16 + 0 * output_stride));
  p1 = _mm_loadl_epi64((const __m128i *)(output_pixels16 + 1 * output_stride));
  p2 = _mm_loadl_epi64((const __m128i *)(output_pixels16 + 2 * output_stride));
  p3 = _mm_loadl_epi64((const __m128i *)(output_pixels16 + 3 * output_stride));
  q0 = _mm_srai_epi16(_mm_add_epi16(q0, round), downshift);
  q1 = _mm_srai_epi16(_mm_add_epi16(q1, round), downshift);
  q2 = _mm_srai_epi16(_mm_add_epi16(q2, round), downshift);
  q3 = _mm_srai_epi16(_mm_add_epi16(q3, round), downshift);
  p0 = od_hbd_clamp_epi16(_mm_add_epi16(p0, q0), max);
  p1 = od_hbd_clamp_epi16(_mm_add_epi16(p1, q1), max);
  p2 = od_hbd_clamp_epi16(_mm_add_epi16(p2, q2), max);
  p3 = od_hbd_clamp_epi16(_mm_add_epi16(p3, q3), max);
  _mm_storel_epi64((__m128i *)(output_pixels16 + 0 * output_stride), p0);
  _mm_storel_epi64((__m128i *)(output_pixels16 + 1 * output_stride), p1);
  _mm_storel_epi64((__m128i *)(output_pixels16 + 2 * output_stride), p2);
  _mm_storel_epi64((__m128i *)(output_pixels16 + 3 * output_stride), p3);
}

/* Loads an 8x4 buffer of 16-bit values, adds a 8x4 block of 16-bit values to
   them, clamps to the high bit depth max, and stores the sum back. */
static INLINE void od_add_store_buffer_hbd_8x4_epi16(void *output_pixels,
                                                     int output_stride,
                                                     __m128i q0, __m128i q1,
                                                     __m128i q2, __m128i q3,
                                                     int bd) {
  uint16_t *output_pixels16;
  __m128i p0;
  __m128i p1;
  __m128i p2;
  __m128i p3;
  __m128i max;
  __m128i round;
  int downshift;
  output_pixels16 = CONVERT_TO_SHORTPTR(output_pixels);
  max = od_hbd_max_epi16(bd);
  downshift = TX_COEFF_DEPTH - bd;
  round = _mm_set1_epi16((1 << downshift) >> 1);
  p0 = _mm_loadu_si128((const __m128i *)(output_pixels16 + 0 * output_stride));
  p1 = _mm_loadu_si128((const __m128i *)(output_pixels16 + 1 * output_stride));
  p2 = _mm_loadu_si128((const __m128i *)(output_pixels16 + 2 * output_stride));
  p3 = _mm_loadu_si128((const __m128i *)(output_pixels16 + 3 * output_stride));
  q0 = _mm_srai_epi16(_mm_add_epi16(q0, round), downshift);
  q1 = _mm_srai_epi16(_mm_add_epi16(q1, round), downshift);
  q2 = _mm_srai_epi16(_mm_add_epi16(q2, round), downshift);
  q3 = _mm_srai_epi16(_mm_add_epi16(q3, round), downshift);
  p0 = od_hbd_clamp_epi16(_mm_add_epi16(p0, q0), max);
  p1 = od_hbd_clamp_epi16(_mm_add_epi16(p1, q1), max);
  p2 = od_hbd_clamp_epi16(_mm_add_epi16(p2, q2), max);
  p3 = od_hbd_clamp_epi16(_mm_add_epi16(p3, q3), max);
  _mm_storeu_si128((__m128i *)(output_pixels16 + 0 * output_stride), p0);
  _mm_storeu_si128((__m128i *)(output_pixels16 + 1 * output_stride), p1);
  _mm_storeu_si128((__m128i *)(output_pixels16 + 2 * output_stride), p2);
  _mm_storeu_si128((__m128i *)(output_pixels16 + 3 * output_stride), p3);
}

static INLINE void od_add_store_buffer_hbd_16x4_epi16(void *output_pixels,
                                                      int output_stride,
                                                      __m256i r0, __m256i r1,
                                                      __m256i r2, __m256i r3,
                                                      int bd) {
  uint16_t *output_pixels16;
  __m256i p0;
  __m256i p1;
  __m256i p2;
  __m256i p3;
  __m256i max;
  __m256i round;
  int downshift;
  output_pixels16 = CONVERT_TO_SHORTPTR(output_pixels);
  max = od_mm256_hbd_max_epi16(bd);
  downshift = TX_COEFF_DEPTH - bd;
  round = _mm256_set1_epi16((1 << downshift) >> 1);
  p0 = _mm256_loadu_si256(
      (const __m256i *)(output_pixels16 + 0 * output_stride));
  p1 = _mm256_loadu_si256(
      (const __m256i *)(output_pixels16 + 1 * output_stride));
  p2 = _mm256_loadu_si256(
      (const __m256i *)(output_pixels16 + 2 * output_stride));
  p3 = _mm256_loadu_si256(
      (const __m256i *)(output_pixels16 + 3 * output_stride));
  r0 = _mm256_srai_epi16(_mm256_add_epi16(r0, round), downshift);
  r1 = _mm256_srai_epi16(_mm256_add_epi16(r1, round), downshift);
  r2 = _mm256_srai_epi16(_mm256_add_epi16(r2, round), downshift);
  r3 = _mm256_srai_epi16(_mm256_add_epi16(r3, round), downshift);
  p0 = od_mm256_hbd_clamp_epi16(_mm256_add_epi16(p0, r0), max);
  p1 = od_mm256_hbd_clamp_epi16(_mm256_add_epi16(p1, r1), max);
  p2 = od_mm256_hbd_clamp_epi16(_mm256_add_epi16(p2, r2), max);
  p3 = od_mm256_hbd_clamp_epi16(_mm256_add_epi16(p3, r3), max);
  _mm256_storeu_si256((__m256i *)(output_pixels16 + 0 * output_stride), p0);
  _mm256_storeu_si256((__m256i *)(output_pixels16 + 1 * output_stride), p1);
  _mm256_storeu_si256((__m256i *)(output_pixels16 + 2 * output_stride), p2);
  _mm256_storeu_si256((__m256i *)(output_pixels16 + 3 * output_stride), p3);
}

static INLINE void od_8x4_cvtepi16_epi32(__m128i *q0, __m128i *q1, __m128i *q2,
                                         __m128i *q3, __m128i *extra) {
  const __m128i zero = _mm_setzero_si128();
  __m128i sign0;
  __m128i sign1;
  __m128i sign2;
  __m128i sign3;
  sign0 = _mm_cmpgt_epi16(zero, *q0);
  sign1 = _mm_cmpgt_epi16(zero, *q1);
  sign2 = _mm_cmpgt_epi16(zero, *q2);
  sign3 = _mm_cmpgt_epi16(zero, *q3);
  extra[0] = _mm_unpackhi_epi16(*q0, sign0);
  extra[1] = _mm_unpackhi_epi16(*q1, sign1);
  extra[2] = _mm_unpackhi_epi16(*q2, sign2);
  extra[3] = _mm_unpackhi_epi16(*q3, sign3);
  *q0 = _mm_unpacklo_epi16(*q0, sign0);
  *q1 = _mm_unpacklo_epi16(*q1, sign1);
  *q2 = _mm_unpacklo_epi16(*q2, sign2);
  *q3 = _mm_unpacklo_epi16(*q3, sign3);
}

static INLINE void od_transpose_pack4x4(__m128i *q0, __m128i *q1, __m128i *q2,
                                        __m128i *q3) {
  __m128i a;
  __m128i b;
  __m128i c;
  __m128i d;
  /* Input:
     q0: q30 q20 q10 q00
     q1: q31 q21 q11 q01
     q2: q32 q22 q12 q02
     q3: q33 q23 q13 q03
  */
  /* a: q32 q22 q12 q02 q30 q20 q10 q00 */
  a = _mm_packs_epi32(*q0, *q2);
  /* b: q33 q23 q13 q03 q31 q21 q11 q01 */
  b = _mm_packs_epi32(*q1, *q3);
  /* c: q31 q30 q21 q20 q11 q10 q01 q00 */
  c = _mm_unpacklo_epi16(a, b);
  /* d: q33 q32 q23 q22 q13 q12 q03 q02 */
  d = _mm_unpackhi_epi16(a, b);
  /* We don't care about the contents of the high half of each register. */
  /* q0: q13 q12 q11 q10 [q03 q02 q01 q00] */
  *q0 = _mm_unpacklo_epi32(c, d);
  /* q1: q13 q12 q11 q10 [q13 q12 q11 q10] */
  *q1 = _mm_unpackhi_epi64(*q0, *q0);
  /* q2: q33 q32 q31 q30 [q23 q22 q21 q20] */
  *q2 = _mm_unpackhi_epi32(c, d);
  /* q3: q33 q32 q31 q30 [q33 q32 q31 q30] */
  *q3 = _mm_unpackhi_epi64(*q2, *q2);
}

static INLINE void od_transpose4x4(__m128i *q0, __m128i q1, __m128i *q2,
                                   __m128i q3) {
  __m128i a;
  __m128i b;
  /* Input:
     q0: ... ... ... ... q30 q20 q10 q00
     q1: ... ... ... ... q31 q21 q11 q01
     q2: ... ... ... ... q32 q22 q12 q02
     q3: ... ... ... ... q33 q23 q13 q03
  */
  /* a: q31 q30 q21 q20 q11 q10 q01 q00 */
  a = _mm_unpacklo_epi16(*q0, q1);
  /* b: q33 q32 q23 q22 q13 q12 q03 q02 */
  b = _mm_unpacklo_epi16(*q2, q3);
  /* q0: q13 q12 q11 q10 | q03 q02 q01 q00 */
  *q0 = _mm_unpacklo_epi32(a, b);
  /* q2: q33 q32 q31 q30 | q23 q22 q21 q20 */
  *q2 = _mm_unpackhi_epi32(a, b);
}

/* Transpose 4x4 16-bit value and then convert them to 32-bits. */
static INLINE void od_transpose_unpack4x4(__m128i *q0, __m128i *q1, __m128i *q2,
                                          __m128i *q3) {
  const __m128i zero = _mm_setzero_si128();
  __m128i a;
  __m128i b;
  __m128i sign0;
  __m128i sign1;
  od_transpose4x4(q0, *q1, q2, *q3);
  a = *q0;
  b = *q2;
  sign0 = _mm_cmpgt_epi16(zero, a);
  sign1 = _mm_cmpgt_epi16(zero, b);
  *q0 = _mm_unpacklo_epi16(a, sign0);
  *q1 = _mm_unpackhi_epi16(a, sign0);
  *q2 = _mm_unpacklo_epi16(b, sign1);
  *q3 = _mm_unpackhi_epi16(b, sign1);
}

static inline void od_transpose4x8(__m128i *r0, __m128i r1, __m128i *r2,
                                   __m128i r3, __m128i *r4, __m128i r5,
                                   __m128i *r6, __m128i r7) {
  __m128i a;
  __m128i b;
  /* Input:
     q0: ... ... ... ... q30 q20 q10 q00
     q1: ... ... ... ... q31 q21 q11 q01
     q2: ... ... ... ... q32 q22 q12 q02
     q3: ... ... ... ... q33 q23 q13 q03
     q4: ... ... ... ... q34 q24 q14 q04
     q5: ... ... ... ... q35 q25 q15 q05
     q6: ... ... ... ... q36 q26 q16 q06
     q7: ... ... ... ... q37 q27 q17 q07
  */
  /* r0: r13 r12 11 r10 r03 r02 r01 r00
     r2: r33 r32 31 r30 r23 r22 r21 r20 */
  od_transpose4x4(r0, r1, r2, r3);
  /* r4: r17 r16 15 r14 r07 r06 r05 r04
     r6: r37 r36 35 r34 r27 r26 r25 r24 */
  od_transpose4x4(r4, r5, r6, r7);
  a = *r0;
  b = *r2;
  /* r0: r07 r06 r05 r04 r04 r02 r01 r00 */
  *r0 = _mm_unpacklo_epi64(a, *r4);
  /* r2: r17 r16 r15 r14 r14 r12 r11 r10 */
  *r2 = _mm_unpackhi_epi64(a, *r4);
  /* r4: r27 r26 r25 r24 r24 r22 r21 r20 */
  *r4 = _mm_unpacklo_epi64(b, *r6);
  /* r6: r37 r36 r35 r34 r34 r32 r31 r30 */
  *r6 = _mm_unpackhi_epi64(b, *r6);
}

static INLINE void od_transpose8x4(__m128i *q0, __m128i *q1, __m128i *q2,
                                   __m128i *q3) {
  __m128i a;
  __m128i b;
  __m128i c;
  __m128i d;
  /* Input:
     q0: q07 q06 q05 q04 q03 q02 q01 q00
     q1: q17 q16 q15 q14 q13 q12 q11 q10
     q2: q27 q26 q25 q24 q23 q22 q21 q20
     q3: q37 q36 q35 q34 q33 q32 q31 q30
  */
  /* a: q13 q03 q12 q02 q11 q01 q10 q00 */
  a = _mm_unpacklo_epi16(*q0, *q1);
  /* b: q17 q07 q16 q06 q15 q05 q14 q04 */
  b = _mm_unpackhi_epi16(*q0, *q1);
  /* c: q33 q23 q32 q22 q31 q21 q30 q20 */
  c = _mm_unpacklo_epi16(*q2, *q3);
  /* d: q37 q27 q36 q26 q35 q25 q34 q24 */
  d = _mm_unpackhi_epi16(*q2, *q3);
  /* q0: q31 q21 q11 q01 | q30 q20 q10 q00 */
  *q0 = _mm_unpacklo_epi32(a, c);
  /* q1: q33 q23 q13 q03 | q32 q22 q12 q02 */
  *q1 = _mm_unpackhi_epi32(a, c);
  /* q2: q35 q25 q15 q05 | q34 q24 q14 q04 */
  *q2 = _mm_unpacklo_epi32(b, d);
  /* q3: q37 q27 q17 q07 | q36 q26 q16 q06 */
  *q3 = _mm_unpackhi_epi32(b, d);
}

static INLINE void od_transpose_pack4x8(__m128i *q0, __m128i *q1, __m128i *q2,
                                        __m128i *q3, __m128i q4, __m128i q5,
                                        __m128i q6, __m128i q7) {
  __m128i a;
  __m128i b;
  __m128i c;
  __m128i d;
  /* Input:
     q0: q30 q20 q10 q00
     q1: q31 q21 q11 q01
     q2: q32 q22 q12 q02
     q3: q33 q23 q13 q03
     q4: q34 q24 q14 q04
     q5: q35 q25 q15 q05
     q6: q36 q26 q16 q06
     q7: q37 q27 q17 q07
  */
  /* a: q34 q24 q14 q04 q30 q20 q10 q00 */
  a = _mm_packs_epi32(*q0, q4);
  /* b: q35 q25 q15 q05 q31 q21 q11 q01 */
  b = _mm_packs_epi32(*q1, q5);
  /* c: q36 q26 q16 q06 q32 q22 q12 q02 */
  c = _mm_packs_epi32(*q2, q6);
  /* d: q37 q27 q17 q07 q33 q23 q13 q03 */
  d = _mm_packs_epi32(*q3, q7);
  /* a: q13 q12 q11 q10 q03 q02 q01 q00
     b: q33 q32 q31 q30 q33 q22 q21 q20
     c: q53 q52 q51 q50 q43 q42 q41 q40
     d: q73 q72 q71 q70 q63 q62 q61 q60 */
  od_transpose8x4(&a, &b, &c, &d);
  /* q0: q07 q06 q05 q04 q03 q02 q01 q00 */
  *q0 = _mm_unpacklo_epi64(a, c);
  /* q1: q17 q16 q15 q14 q13 q12 q11 q10 */
  *q1 = _mm_unpackhi_epi64(a, c);
  /* q2: q27 q26 q25 q24 q23 q22 q21 q20 */
  *q2 = _mm_unpacklo_epi64(b, d);
  /* q3: q37 q36 q35 q34 q33 q32 q31 q30 */
  *q3 = _mm_unpackhi_epi64(b, d);
}

static INLINE void od_transpose_pack8x4(__m128i *r0, __m128i *r1, __m128i *r2,
                                        __m128i *r3, __m128i *r4, __m128i *r5,
                                        __m128i *r6, __m128i *r7) {
  /* Input:
     r1: r07 r06 r05 r04  r0: r03 r02 r01 r00
     r3: r17 r16 r15 r14  r2: r13 r12 r11 r10
     r5: r27 r26 r25 r24  r4: r23 r22 r21 r20
     r7: r37 r36 r35 r34  r6: r33 r32 r31 r30
  */
  /* r0: r07 r06 r05 r04 r03 r02 r01 r00 */
  *r0 = _mm_packs_epi32(*r0, *r1);
  /* r2: r17 r16 r15 r14 r13 r12 r11 r10 */
  *r2 = _mm_packs_epi32(*r2, *r3);
  /* r4: r27 r26 r25 r24 r23 r22 r21 r20 */
  *r4 = _mm_packs_epi32(*r4, *r5);
  /* r6: r37 r36 r35 r34 r33 r32 r31 r30 */
  *r6 = _mm_packs_epi32(*r6, *r7);
  /* r0: r31 r21 r11 r01 [r30 r20 r10 r00]
     r2: r33 r23 r13 r03 [r32 r22 r12 r02]
     r4: r35 r25 r15 r05 [r34 r24 r14 r04]
     r6: r37 r27 r17 r07 [r36 r26 r16 r06] */
  od_transpose8x4(r0, r2, r4, r6);
  /* We don't care about the contents of the high half of each register. */
  /* r1: r31 r21 r11 r01 [r31 r21 r11 r01] */
  *r1 = _mm_unpackhi_epi64(*r0, *r0);
  /* r3: r33 r23 r13 r03 [r33 r23 r13 r03] */
  *r3 = _mm_unpackhi_epi64(*r2, *r2);
  /* r5: r35 r25 r15 r05 [r35 r25 r15 r05] */
  *r5 = _mm_unpackhi_epi64(*r4, *r4);
  /* r7: r37 r27 r17 r07 [r37 r27 r17 r07] */
  *r7 = _mm_unpackhi_epi64(*r6, *r6);
}

static INLINE void od_transpose8x8_epi16(__m128i *r0, __m128i *r1, __m128i *r2,
                                         __m128i *r3, __m128i *r4, __m128i *r5,
                                         __m128i *r6, __m128i *r7) {
  __m128i r8;
  /*8x8 transpose with only 1 temporary register that takes the rows in order
    and returns the columns in order. The compiler's own register allocator
    will probably screw this up, but that's no reason not to pretend we might
    be able to have nice things. This only matters when we port to pre-AVX
    instruction sets without 3-operand instructions.*/
  r8 = *r4;
  *r4 = _mm_unpacklo_epi16(*r4, *r5);
  r8 = _mm_unpackhi_epi16(r8, *r5);
  *r5 = *r0;
  *r0 = _mm_unpacklo_epi16(*r0, *r1);
  *r5 = _mm_unpackhi_epi16(*r5, *r1);
  *r1 = *r6;
  *r6 = _mm_unpacklo_epi16(*r6, *r7);
  *r1 = _mm_unpackhi_epi16(*r1, *r7);
  *r7 = *r2;
  *r2 = _mm_unpackhi_epi16(*r2, *r3);
  *r7 = _mm_unpacklo_epi16(*r7, *r3);
  *r3 = *r0;
  *r0 = _mm_unpacklo_epi32(*r0, *r7);
  *r3 = _mm_unpackhi_epi32(*r3, *r7);
  *r7 = *r5;
  *r5 = _mm_unpacklo_epi32(*r5, *r2);
  *r7 = _mm_unpackhi_epi32(*r7, *r2);
  *r2 = *r4;
  *r4 = _mm_unpackhi_epi32(*r4, *r6);
  *r2 = _mm_unpacklo_epi32(*r2, *r6);
  *r6 = r8;
  r8 = _mm_unpackhi_epi32(r8, *r1);
  *r6 = _mm_unpacklo_epi32(*r6, *r1);
  *r1 = *r0;
  *r0 = _mm_unpacklo_epi64(*r0, *r2);
  *r1 = _mm_unpackhi_epi64(*r1, *r2);
  *r2 = *r3;
  *r3 = _mm_unpackhi_epi64(*r3, *r4);
  *r2 = _mm_unpacklo_epi64(*r2, *r4);
  *r4 = *r5;
  *r5 = _mm_unpackhi_epi64(*r5, *r6);
  *r4 = _mm_unpacklo_epi64(*r4, *r6);
  *r6 = *r7;
  *r7 = _mm_unpackhi_epi64(*r7, r8);
  *r6 = _mm_unpacklo_epi64(*r6, r8);
}

static INLINE void od_transpose8x8_epi32(__m256i *r0, __m256i *r1, __m256i *r2,
                                         __m256i *r3, __m256i *r4, __m256i *r5,
                                         __m256i *r6, __m256i *r7) {
  __m256i a;
  __m256i b;
  __m256i c;
  __m256i d;
  __m256i e;
  __m256i f;
  __m256i g;
  __m256i h;
  __m256i x;
  __m256i y;
  a = _mm256_unpacklo_epi32(*r0, *r1);
  b = _mm256_unpacklo_epi32(*r2, *r3);
  c = _mm256_unpackhi_epi32(*r0, *r1);
  d = _mm256_unpackhi_epi32(*r2, *r3);
  e = _mm256_unpacklo_epi32(*r4, *r5);
  f = _mm256_unpacklo_epi32(*r6, *r7);
  g = _mm256_unpackhi_epi32(*r4, *r5);
  h = _mm256_unpackhi_epi32(*r6, *r7);
  x = _mm256_unpacklo_epi64(a, b);
  y = _mm256_unpacklo_epi64(e, f);
  *r0 = _mm256_permute2x128_si256(x, y, 0 | (2 << 4));
  *r4 = _mm256_permute2x128_si256(x, y, 1 | (3 << 4));
  x = _mm256_unpackhi_epi64(a, b);
  y = _mm256_unpackhi_epi64(e, f);
  *r1 = _mm256_permute2x128_si256(x, y, 0 | (2 << 4));
  *r5 = _mm256_permute2x128_si256(x, y, 1 | (3 << 4));
  x = _mm256_unpacklo_epi64(c, d);
  y = _mm256_unpacklo_epi64(g, h);
  *r2 = _mm256_permute2x128_si256(x, y, 0 | (2 << 4));
  *r6 = _mm256_permute2x128_si256(x, y, 1 | (3 << 4));
  x = _mm256_unpackhi_epi64(c, d);
  y = _mm256_unpackhi_epi64(g, h);
  *r3 = _mm256_permute2x128_si256(x, y, 0 | (2 << 4));
  *r7 = _mm256_permute2x128_si256(x, y, 1 | (3 << 4));
}

/* Packs two blocks of 4x8 32-bit words into 16-bit words and returns the
   transpose of each packed into the high and low halves of each register. */
static INLINE void od_transpose_pack4x8x2_epi32(__m256i *out0, __m256i *out1,
                                                __m256i *out2, __m256i *out3,
                                                __m256i rr0, __m256i rr1,
                                                __m256i rr2, __m256i rr3,
                                                __m256i rr4, __m256i rr5,
                                                __m256i rr6, __m256i rr7) {
  __m256i a;
  __m256i b;
  __m256i c;
  __m256i d;
  __m256i w;
  __m256i x;
  __m256i y;
  __m256i z;
  /* a: r47 r46 r45 r44 r07 r06 r05 r04 | r43 r42 r41 r40 r03 r02 r01 r00 */
  a = _mm256_packs_epi32(rr0, rr4);
  /* b: r57 r56 r55 r54 r17 r16 r15 r14 | r53 r52 r51 r50 r13 r12 r11 r10 */
  b = _mm256_packs_epi32(rr1, rr5);
  /* c: r67 r66 r65 r64 r27 r26 r25 r24 | r63 r62 r61 r60 r23 r22 r21 r20 */
  c = _mm256_packs_epi32(rr2, rr6);
  /* d: r77 r76 r75 r74 r37 r36 r35 r34 | r73 r72 r71 r70 r33 r32 r31 r30 */
  d = _mm256_packs_epi32(rr3, rr7);
  /* w: r17 r07 r16 r06 r15 r05 r14 r04 | r13 r03 r12 r02 r11 r01 r10 r00 */
  w = _mm256_unpacklo_epi16(a, b);
  /* x: r57 r47 r56 r46 r55 r45 r54 r44 | r53 r43 r52 r42 r51 r41 r50 r40 */
  x = _mm256_unpackhi_epi16(a, b);
  /* y: r37 r27 r36 r26 r35 r25 r34 r24 | r33 r23 r32 r22 r31 r21 r30 r20 */
  y = _mm256_unpacklo_epi16(c, d);
  /* z: r77 r67 r76 r66 r75 r65 r74 r64 | r73 r63 r72 r62 r71 r61 r70 r60 */
  z = _mm256_unpackhi_epi16(c, d);
  /* a: r35 r25 r15 r05 r34 r24 r14 r04 | r31 r21 r11 r01 r30 r20 r10 r00 */
  a = _mm256_unpacklo_epi32(w, y);
  /* b: r77 r67 r57 r47 r76 r66 r56 r46 | r33 r23 r13 r03 r32 r22 r12 r02 */
  b = _mm256_unpackhi_epi32(w, y);
  /* c: r75 r65 r55 r45 r74 r64 r54 r44 | r71 r61 r51 r41 r70 r60 r50 r40 */
  c = _mm256_unpacklo_epi32(x, z);
  /* d: r77 r67 r57 r47 r76 r66 r56 r46 | r73 r63 r53 r43 r72 r62 r52 r42 */
  d = _mm256_unpackhi_epi32(x, z);
  /* out0: r74 r64 r54 r44 r34 r24 r14 r04 | r70 r60 r50 r40 r30 r20 r10 r00 */
  *out0 = _mm256_unpacklo_epi64(a, c);
  /* out1: r75 r65 r55 r45 r35 r25 r15 r05 | r71 r61 r51 r41 r31 r21 r11 r01 */
  *out1 = _mm256_unpackhi_epi64(a, c);
  /* out2: r76 r66 r56 r46 r36 r26 r16 r06 | r72 r62 r52 r42 r32 r22 r12 r02 */
  *out2 = _mm256_unpacklo_epi64(b, d);
  /* out3: r77 r67 r57 r47 r37 r27 r17 r07 | r73 r63 r53 r43 r33 r23 r13 r03 */
  *out3 = _mm256_unpackhi_epi64(b, d);
}

static INLINE void od_transpose_pack8x8_epi32(__m256i *rr0, __m256i *rr1,
                                              __m256i *rr2, __m256i *rr3,
                                              __m256i rr4, __m256i rr5,
                                              __m256i rr6, __m256i rr7) {
  __m256i w;
  __m256i x;
  __m256i y;
  __m256i z;
  /* w: r74 r64 r54 r44 r34 r24 r14 r04 | r70 r60 r50 r40 r30 r20 r10 r00
     x: r75 r65 r55 r45 r35 r25 r15 r05 | r71 r61 r51 r41 r31 r21 r11 r01
     y: r76 r66 r56 r46 r36 r26 r16 r06 | r72 r62 r52 r42 r32 r22 r12 r02
     z: r77 r67 r57 r47 r37 r27 r17 r07 | r73 r63 r53 r43 r33 r23 r13 r03 */
  od_transpose_pack4x8x2_epi32(&w, &x, &y, &z, *rr0, *rr1, *rr2, *rr3, rr4, rr5,
                               rr6, rr7);
  /* rr0: r71 r61 r51 r41 r31 r21 r11 r01 | r70 r60 r50 r40 r30 r20 r10 r00 */
  *rr0 = _mm256_permute2x128_si256(w, x, 0 | (2 << 4));
  /* rr1: r73 r63 r53 r43 r33 r23 r13 r03 | r72 r62 r52 r42 r32 r22 r12 r02 */
  *rr1 = _mm256_permute2x128_si256(y, z, 0 | (2 << 4));
  /* rr2: r75 r65 r55 r45 r35 r25 r15 r05 r74 r64 r54 r44 r34 r24 r14 r04 */
  *rr2 = _mm256_permute2x128_si256(w, x, 1 | (3 << 4));
  /* rr3: r77 r67 r57 r47 r37 r27 r17 r07 r76 r66 r56 r46 r36 r26 r16 r06 */
  *rr3 = _mm256_permute2x128_si256(y, z, 1 | (3 << 4));
}

static INLINE void od_transpose_pack8x16_epi32(
    __m256i *ss0, __m256i *ss1, __m256i *ss2, __m256i *ss3, __m256i *ss4,
    __m256i *ss5, __m256i *ss6, __m256i *ss7, __m256i ss8, __m256i ss9,
    __m256i ssa, __m256i ssb, __m256i ssc, __m256i ssd, __m256i sse,
    __m256i ssf) {
  __m256i a;
  __m256i b;
  __m256i c;
  __m256i d;
  __m256i e;
  __m256i f;
  __m256i g;
  __m256i h;
  /* ss0: s74 s64 s54 s44 s34 s24 s14 s04 | s70 s60 s50 s40 s30 s20 s10 s00
     ss2: s75 s65 s55 s45 s35 s25 s15 s05 | s71 s61 s51 s41 s31 s21 s11 s01
     ss4: s76 s66 s56 s46 s36 s26 s16 s06 | s72 s62 s52 s42 s32 s22 s12 s02
     ss6: s77 s67 s57 s47 s37 s27 s17 s07 | s73 s63 s53 s43 s33 s23 s13 s03 */
  od_transpose_pack4x8x2_epi32(&a, &b, &c, &d, *ss0, *ss1, *ss2, *ss3, *ss4,
                               *ss5, *ss6, *ss7);
  /* ss8: sf4 se4 sd4 sc4 sb4 sa4 s94 s84 | sf0 se0 sd0 sc0 sb0 sa0 s90 s80
     ssa: sf5 se5 sd5 sc5 sb5 sa5 s95 s85 | sf1 se1 sd1 sc1 sb1 sa1 s91 s81
     ssc: sf6 se6 sd6 sc6 sb6 sa6 s96 s86 | sf2 se2 sd2 sc2 sb2 sa2 s92 s82
     sse: sf7 se7 sd7 sc7 sb7 sa7 s97 s87 | sf3 se3 sd3 sc3 sb3 sa3 s93 s83 */
  od_transpose_pack4x8x2_epi32(&e, &f, &g, &h, ss8, ss9, ssa, ssb, ssc, ssd,
                               sse, ssf);
  /* ss0: sf0 se0 sd0 sc0 sb0 sa0 s90 s80 | s70 s60 s50 s40 s30 s20 s10 s00 */
  *ss0 = _mm256_permute2x128_si256(a, e, 0 | (2 << 4));
  /* ss1: sf1 se1 sd1 sc1 sb1 sa1 s91 s81 | s71 s61 s51 s41 s31 s21 s11 s01 */
  *ss1 = _mm256_permute2x128_si256(b, f, 0 | (2 << 4));
  /* ss2: sf2 se2 sd2 sc2 sb2 sa2 s92 s82 | s72 s62 s52 s42 s32 s22 s12 s02 */
  *ss2 = _mm256_permute2x128_si256(c, g, 0 | (2 << 4));
  /* ss3: sf3 se3 sd3 sc3 sb3 sa3 s93 s83 | s73 s63 s53 s43 s33 s23 s13 s03 */
  *ss3 = _mm256_permute2x128_si256(d, h, 0 | (2 << 4));
  /* ss4: sf4 se4 sd4 sc4 sb4 sa4 s94 s84 | s74 s64 s54 s44 s34 s24 s14 s04 */
  *ss4 = _mm256_permute2x128_si256(a, e, 1 | (3 << 4));
  /* ss5: sf5 se5 sd5 sc5 sb5 sa5 s95 s85 | s75 s65 s55 s45 s35 s25 s15 s05 */
  *ss5 = _mm256_permute2x128_si256(b, f, 1 | (3 << 4));
  /* ss6: rf6 re6 rd6 rc6 rb6 ra6 r96 r82 | r76 r66 r56 r46 r36 r26 r16 r06 */
  *ss6 = _mm256_permute2x128_si256(c, g, 1 | (3 << 4));
  /* ss7: rf7 re7 rd7 rc7 rb7 ra7 r97 r87 | r77 r67 r57 r47 r37 r27 r17 r07 */
  *ss7 = _mm256_permute2x128_si256(d, h, 1 | (3 << 4));
}

#undef OD_KERNEL
#undef OD_WORD
#undef OD_REG
#undef OD_ADD
#undef OD_SUB
#undef OD_RSHIFT1
#undef OD_AVG
#undef OD_HRSUB
#undef OD_MUL
#undef OD_SWAP

/* Define 8-wide 16-bit SSSE3 kernels. */

#define OD_KERNEL kernel8
#define OD_WORD epi16
#define OD_REG __m128i
#define OD_ADD _mm_add_epi16
#define OD_SUB _mm_sub_epi16
#define OD_RSHIFT1 od_unbiased_rshift1_epi16
#define OD_AVG od_avg_epi16
#define OD_HRSUB od_hrsub_epi16
#define OD_MUL od_mul_epi16
#define OD_SWAP od_swap_si128

#include "av1/common/x86/daala_tx_kernels.h"

#undef OD_KERNEL
#undef OD_REG
#undef OD_ADD
#undef OD_SUB
#undef OD_RSHIFT1
#undef OD_AVG
#undef OD_HRSUB
#undef OD_MUL
#undef OD_SWAP

/* Define 16-wide 16-bit AVX2 kernels. */

#define OD_KERNEL kernel16
#define OD_REG __m256i
#define OD_ADD _mm256_add_epi16
#define OD_SUB _mm256_sub_epi16
#define OD_RSHIFT1 od_mm256_unbiased_rshift1_epi16
#define OD_AVG od_mm256_avg_epi16
#define OD_HRSUB od_mm256_hrsub_epi16
#define OD_MUL od_mm256_mul_epi16
#define OD_SWAP od_mm256_swap_si256

#include "av1/common/x86/daala_tx_kernels.h"  // NOLINT

/* Define 8-wide 32-bit AVX2 kernels. */

#undef OD_KERNEL
#undef OD_WORD
#undef OD_ADD
#undef OD_SUB
#undef OD_RSHIFT1
#undef OD_AVG
#undef OD_HRSUB
#undef OD_MUL

#define OD_KERNEL kernel8
#define OD_WORD epi32
#define OD_ADD _mm256_add_epi32
#define OD_SUB _mm256_sub_epi32
#define OD_RSHIFT1 od_mm256_unbiased_rshift1_epi32
#define OD_AVG od_mm256_avg_epi32
#define OD_HRSUB od_mm256_hrsub_epi32
#define OD_MUL od_mm256_mul_epi32

#include "av1/common/x86/daala_tx_kernels.h"  // NOLINT

#endif

#endif  // DAALA_TX_AVX2_H_
