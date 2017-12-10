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

#include <tmmintrin.h>
#include <immintrin.h>
#include "./av1_rtcd.h"
#include "./aom_config.h"
#include "./aom_dsp_rtcd.h"
#include "av1/common/daala_tx.h"
#include "av1/common/daala_inv_txfm.h"
#include "av1/common/idct.h"
#include "av1/common/x86/daala_tx_avx2.h"

#if CONFIG_DAALA_TX

static void od_row_iidtx_avx2(int16_t *out, int coeffs, const tran_low_t *in) {
  int c;
  /* The number of rows and number of columns are both multiples of 4, so the
     total number of coefficients should be a multiple of 16. */
  assert(!(coeffs & 0xF));
  /* TODO(any): Use AVX2 for larger block sizes. */
  for (c = 0; c < coeffs; c += 16) {
    __m128i q0;
    __m128i q1;
    __m128i q2;
    __m128i q3;
    od_load_buffer_4x4_epi32(&q0, &q1, &q2, &q3, in + c);
    q0 = _mm_packs_epi32(q0, q1);
    q2 = _mm_packs_epi32(q2, q3);
    od_store_buffer_4x4_epi16(out + c, q0, q2);
  }
}

static void od_col_iidtx_add_hbd_avx2(unsigned char *output_pixels,
                                      int output_stride, int rows, int cols,
                                      const int16_t *in, int bd) {
  __m128i q0;
  __m128i q1;
  __m128i q2;
  __m128i q3;
  if (cols <= 4) {
    uint16_t *output_pixels16;
    __m128i p0;
    __m128i p1;
    __m128i p2;
    __m128i p3;
    __m128i max;
    __m128i round;
    int downshift;
    int hr;
    output_pixels16 = CONVERT_TO_SHORTPTR(output_pixels);
    max = od_hbd_max_epi16(bd);
    downshift = TX_COEFF_DEPTH - bd;
    round = _mm_set1_epi16((1 << downshift) >> 1);
    /* Here hr counts half the number of rows, to simplify address calculations
       when loading two rows of coefficients at once. */
    for (hr = 0; 2 * hr < rows; hr += 2) {
      q0 = _mm_loadu_si128((const __m128i *)in + hr + 0);
      q2 = _mm_loadu_si128((const __m128i *)in + hr + 1);
      p0 = _mm_loadl_epi64(
          (const __m128i *)(output_pixels16 + (2 * hr + 0) * output_stride));
      p1 = _mm_loadl_epi64(
          (const __m128i *)(output_pixels16 + (2 * hr + 1) * output_stride));
      p2 = _mm_loadl_epi64(
          (const __m128i *)(output_pixels16 + (2 * hr + 2) * output_stride));
      p3 = _mm_loadl_epi64(
          (const __m128i *)(output_pixels16 + (2 * hr + 3) * output_stride));
      q0 = _mm_srai_epi16(_mm_add_epi16(q0, round), downshift);
      q2 = _mm_srai_epi16(_mm_add_epi16(q2, round), downshift);
      q1 = _mm_unpackhi_epi64(q0, q0);
      q3 = _mm_unpackhi_epi64(q2, q2);
      p0 = od_hbd_clamp_epi16(_mm_add_epi16(p0, q0), max);
      p1 = od_hbd_clamp_epi16(_mm_add_epi16(p1, q1), max);
      p2 = od_hbd_clamp_epi16(_mm_add_epi16(p2, q2), max);
      p3 = od_hbd_clamp_epi16(_mm_add_epi16(p3, q3), max);
      _mm_storel_epi64(
          (__m128i *)(output_pixels16 + (2 * hr + 0) * output_stride), p0);
      _mm_storel_epi64(
          (__m128i *)(output_pixels16 + (2 * hr + 1) * output_stride), p1);
      _mm_storel_epi64(
          (__m128i *)(output_pixels16 + (2 * hr + 2) * output_stride), p2);
      _mm_storel_epi64(
          (__m128i *)(output_pixels16 + (2 * hr + 3) * output_stride), p3);
    }
  } else {
    int r;
    for (r = 0; r < rows; r += 4) {
      int c;
      /* TODO(any): Use AVX2 for larger column counts. */
      for (c = 0; c < cols; c += 8) {
        od_load_buffer_8x4_epi16(&q0, &q1, &q2, &q3, in + r * cols + c, cols);
        od_add_store_buffer_hbd_8x4_epi16(output_pixels + r * output_stride + c,
                                          output_stride, q0, q1, q2, q3, bd);
      }
    }
  }
}

typedef void (*od_tx4_kernel8_epi16)(__m128i *q0, __m128i *q2, __m128i *q1,
                                     __m128i *q3);

static void od_row_tx4_avx2(int16_t *out, int rows, const tran_low_t *in,
                            od_tx4_kernel8_epi16 kernel8) {
  __m128i q0;
  __m128i q1;
  __m128i q2;
  __m128i q3;
  if (rows <= 4) {
    od_load_buffer_4x4_epi32(&q0, &q1, &q2, &q3, in);
    /*TODO(any): Merge this transpose with coefficient scanning.*/
    od_transpose_pack4x4(&q0, &q1, &q2, &q3);
    kernel8(&q0, &q1, &q2, &q3);
    od_transpose4x4(&q0, q2, &q1, q3);
    od_store_buffer_4x4_epi16(out, q0, q1);
  } else {
    int r;
    /* Higher row counts require 32-bit precision. */
    assert(rows <= 16);
    for (r = 0; r < rows; r += 8) {
      __m128i q4;
      __m128i q5;
      __m128i q6;
      __m128i q7;
      od_load_buffer_4x4_epi32(&q0, &q1, &q2, &q3, in + 4 * r);
      od_load_buffer_4x4_epi32(&q4, &q5, &q6, &q7, in + 4 * r + 16);
      /*TODO(any): Merge this transpose with coefficient scanning.*/
      od_transpose_pack4x8(&q0, &q1, &q2, &q3, q4, q5, q6, q7);
      kernel8(&q0, &q1, &q2, &q3);
      od_transpose8x4(&q0, &q2, &q1, &q3);
      od_store_buffer_4x8_epi16(out + 4 * r, q0, q2, q1, q3);
    }
  }
}

static void od_col_tx4_add_hbd_avx2(unsigned char *output_pixels,
                                    int output_stride, int cols,
                                    const int16_t *in, int bd,
                                    od_tx4_kernel8_epi16 kernel8) {
  __m128i q0;
  __m128i q1;
  __m128i q2;
  __m128i q3;
  if (cols <= 4) {
    od_load_buffer_4x4_epi16(&q0, &q1, &q2, &q3, in);
    kernel8(&q0, &q1, &q2, &q3);
    od_add_store_buffer_hbd_4x4_epi16(output_pixels, output_stride, q0, q2, q1,
                                      q3, bd);
  } else {
    int c;
    for (c = 0; c < cols; c += 8) {
      od_load_buffer_8x4_epi16(&q0, &q1, &q2, &q3, in + c, cols);
      kernel8(&q0, &q1, &q2, &q3);
      od_add_store_buffer_hbd_8x4_epi16(output_pixels + c, output_stride, q0,
                                        q2, q1, q3, bd);
    }
  }
}

static void od_row_idct4_avx2(int16_t *out, int rows, const tran_low_t *in) {
  od_row_tx4_avx2(out, rows, in, od_idct4_kernel8_epi16);
}

static void od_col_idct4_add_hbd_avx2(unsigned char *output_pixels,
                                      int output_stride, int cols,
                                      const int16_t *in, int bd) {
  od_col_tx4_add_hbd_avx2(output_pixels, output_stride, cols, in, bd,
                          od_idct4_kernel8_epi16);
}

static void od_row_idst4_avx2(int16_t *out, int rows, const tran_low_t *in) {
  od_row_tx4_avx2(out, rows, in, od_idst_vii4_kernel8_epi16);
}

static void od_col_idst4_add_hbd_avx2(unsigned char *output_pixels,
                                      int output_stride, int cols,
                                      const int16_t *in, int bd) {
  od_col_tx4_add_hbd_avx2(output_pixels, output_stride, cols, in, bd,
                          od_idst_vii4_kernel8_epi16);
}

static void od_row_flip_idst4_avx2(int16_t *out, int rows,
                                   const tran_low_t *in) {
  od_row_tx4_avx2(out, rows, in, od_flip_idst_vii4_kernel8_epi16);
}

static void od_col_flip_idst4_add_hbd_avx2(unsigned char *output_pixels,
                                           int output_stride, int cols,
                                           const int16_t *in, int bd) {
  od_col_tx4_add_hbd_avx2(output_pixels, output_stride, cols, in, bd,
                          od_flip_idst_vii4_kernel8_epi16);
}

static void od_row_iidtx4_avx2(int16_t *out, int rows, const tran_low_t *in) {
  od_row_iidtx_avx2(out, rows * 4, in);
}

static void od_col_iidtx4_add_hbd_avx2(unsigned char *output_pixels,
                                       int output_stride, int cols,
                                       const int16_t *in, int bd) {
  od_col_iidtx_add_hbd_avx2(output_pixels, output_stride, 4, cols, in, bd);
}

typedef void (*od_tx8_kernel8_epi16)(__m128i *r0, __m128i *r4, __m128i *r2,
                                     __m128i *r6, __m128i *r1, __m128i *r5,
                                     __m128i *r3, __m128i *r7);

typedef void (*od_tx8_mm256_kernel)(__m256i *r0, __m256i *r4, __m256i *r2,
                                    __m256i *r6, __m256i *r1, __m256i *r5,
                                    __m256i *r3, __m256i *r7);

static void od_row_tx8_avx2(int16_t *out, int rows, const tran_low_t *in,
                            od_tx8_kernel8_epi16 kernel8_epi16,
                            od_tx8_mm256_kernel kernel8_epi32) {
  __m128i r0;
  __m128i r1;
  __m128i r2;
  __m128i r3;
  __m128i r4;
  __m128i r5;
  __m128i r6;
  __m128i r7;
  if (rows <= 4) {
    od_load_buffer_4x4_epi32(&r0, &r1, &r2, &r3, in);
    od_load_buffer_4x4_epi32(&r4, &r5, &r6, &r7, in + 16);
    /*TODO(any): Merge this transpose with coefficient scanning.*/
    od_transpose_pack8x4(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    kernel8_epi16(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    od_transpose4x8(&r0, r4, &r2, r6, &r1, r5, &r3, r7);
    od_store_buffer_4x4_epi16(out, r0, r2);
    od_store_buffer_4x4_epi16(out + 16, r1, r3);
  } else if (rows <= 8) {
    od_load_pack_buffer_8x4_epi32(&r0, &r1, &r2, &r3, in);
    od_load_pack_buffer_8x4_epi32(&r4, &r5, &r6, &r7, in + 32);
    /*TODO(any): Merge this transpose with coefficient scanning.*/
    od_transpose8x8_epi16(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    kernel8_epi16(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    od_transpose8x8_epi16(&r0, &r4, &r2, &r6, &r1, &r5, &r3, &r7);
    od_store_buffer_4x8_epi16(out, r0, r4, r2, r6);
    od_store_buffer_4x8_epi16(out + 32, r1, r5, r3, r7);
  } else {
    int r;
    /* 16 or more rows requires 32-bit precision.
       TODO(any): If the column TX is IDTX, then we can still use 16 bits. */
    for (r = 0; r < rows; r += 8) {
      __m256i rr0;
      __m256i rr1;
      __m256i rr2;
      __m256i rr3;
      __m256i rr4;
      __m256i rr5;
      __m256i rr6;
      __m256i rr7;
      od_load_buffer_8x4_epi32(&rr0, &rr1, &rr2, &rr3, in + r * 8);
      od_load_buffer_8x4_epi32(&rr4, &rr5, &rr6, &rr7, in + r * 8 + 32);
      od_transpose8x8_epi32(&rr0, &rr1, &rr2, &rr3, &rr4, &rr5, &rr6, &rr7);
      kernel8_epi32(&rr0, &rr1, &rr2, &rr3, &rr4, &rr5, &rr6, &rr7);
      od_transpose_pack8x8_epi32(&rr0, &rr4, &rr2, &rr6, rr1, rr5, rr3, rr7);
      od_store_buffer_2x16_epi16(out + r * 8, rr0, rr4);
      od_store_buffer_2x16_epi16(out + r * 8 + 32, rr2, rr6);
    }
  }
}

static void od_col_tx8_add_hbd_avx2(unsigned char *output_pixels,
                                    int output_stride, int cols,
                                    const int16_t *in, int bd,
                                    od_tx8_kernel8_epi16 kernel8_epi16,
                                    od_tx8_mm256_kernel kernel16_epi16) {
  __m128i r0;
  __m128i r1;
  __m128i r2;
  __m128i r3;
  __m128i r4;
  __m128i r5;
  __m128i r6;
  __m128i r7;
  if (cols <= 4) {
    od_load_buffer_4x4_epi16(&r0, &r1, &r2, &r3, in);
    od_load_buffer_4x4_epi16(&r4, &r5, &r6, &r7, in + 16);
    kernel8_epi16(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    od_add_store_buffer_hbd_4x4_epi16(output_pixels, output_stride, r0, r4, r2,
                                      r6, bd);
    od_add_store_buffer_hbd_4x4_epi16(output_pixels + 4 * output_stride,
                                      output_stride, r1, r5, r3, r7, bd);
  } else if (cols <= 8) {
    od_load_buffer_8x4_epi16(&r0, &r1, &r2, &r3, in, cols);
    od_load_buffer_8x4_epi16(&r4, &r5, &r6, &r7, in + 32, cols);
    kernel8_epi16(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    od_add_store_buffer_hbd_8x4_epi16(output_pixels, output_stride, r0, r4, r2,
                                      r6, bd);
    od_add_store_buffer_hbd_8x4_epi16(output_pixels + 4 * output_stride,
                                      output_stride, r1, r5, r3, r7, bd);
  } else {
    __m256i rr0;
    __m256i rr1;
    __m256i rr2;
    __m256i rr3;
    __m256i rr4;
    __m256i rr5;
    __m256i rr6;
    __m256i rr7;
    int c;
    for (c = 0; c < cols; c += 16) {
      od_load_buffer_16x4_epi16(&rr0, &rr1, &rr2, &rr3, in + c, cols);
      od_load_buffer_16x4_epi16(&rr4, &rr5, &rr6, &rr7, in + 4 * cols + c,
                                cols);
      kernel16_epi16(&rr0, &rr1, &rr2, &rr3, &rr4, &rr5, &rr6, &rr7);
      od_add_store_buffer_hbd_16x4_epi16(output_pixels, output_stride, rr0, rr4,
                                         rr2, rr6, bd);
      od_add_store_buffer_hbd_16x4_epi16(output_pixels + 4 * output_stride,
                                         output_stride, rr1, rr5, rr3, rr7, bd);
    }
  }
}

static void od_row_idct8_avx2(int16_t *out, int rows, const tran_low_t *in) {
  od_row_tx8_avx2(out, rows, in, od_idct8_kernel8_epi16,
                  od_idct8_kernel8_epi32);
}

static void od_col_idct8_add_hbd_avx2(unsigned char *output_pixels,
                                      int output_stride, int cols,
                                      const int16_t *in, int bd) {
  od_col_tx8_add_hbd_avx2(output_pixels, output_stride, cols, in, bd,
                          od_idct8_kernel8_epi16, od_idct8_kernel16_epi16);
}

static void od_row_idst8_avx2(int16_t *out, int rows, const tran_low_t *in) {
  od_row_tx8_avx2(out, rows, in, od_idst8_kernel8_epi16,
                  od_idst8_kernel8_epi32);
}

static void od_col_idst8_add_hbd_avx2(unsigned char *output_pixels,
                                      int output_stride, int cols,
                                      const int16_t *in, int bd) {
  od_col_tx8_add_hbd_avx2(output_pixels, output_stride, cols, in, bd,
                          od_idst8_kernel8_epi16, od_idst8_kernel16_epi16);
}

static void od_row_flip_idst8_avx2(int16_t *out, int rows,
                                   const tran_low_t *in) {
  od_row_tx8_avx2(out, rows, in, od_flip_idst8_kernel8_epi16,
                  od_flip_idst8_kernel8_epi32);
}

static void od_col_flip_idst8_add_hbd_avx2(unsigned char *output_pixels,
                                           int output_stride, int cols,
                                           const int16_t *in, int bd) {
  od_col_tx8_add_hbd_avx2(output_pixels, output_stride, cols, in, bd,
                          od_flip_idst8_kernel8_epi16,
                          od_flip_idst8_kernel16_epi16);
}

static void od_row_iidtx8_avx2(int16_t *out, int rows, const tran_low_t *in) {
  od_row_iidtx_avx2(out, rows * 8, in);
}

static void od_col_iidtx8_add_hbd_avx2(unsigned char *output_pixels,
                                       int output_stride, int cols,
                                       const int16_t *in, int bd) {
  od_col_iidtx_add_hbd_avx2(output_pixels, output_stride, 8, cols, in, bd);
}

typedef void (*od_tx16_kernel8_epi16)(__m128i *s0, __m128i *s4, __m128i *s2,
                                      __m128i *s6, __m128i *s1, __m128i *s5,
                                      __m128i *s3, __m128i *s7, __m128i *s8,
                                      __m128i *s9, __m128i *sa, __m128i *sb,
                                      __m128i *sc, __m128i *sd, __m128i *se,
                                      __m128i *sf);

typedef void (*od_tx16_mm256_kernel)(__m256i *s0, __m256i *s4, __m256i *s2,
                                     __m256i *s6, __m256i *s1, __m256i *s5,
                                     __m256i *s3, __m256i *s7, __m256i *s8,
                                     __m256i *s9, __m256i *sa, __m256i *sb,
                                     __m256i *sc, __m256i *sd, __m256i *se,
                                     __m256i *sf);

static void od_row_tx16_avx2(int16_t *out, int rows, const tran_low_t *in,
#if CONFIG_RECT_TX_EXT
                             od_tx16_kernel8_epi16 kernel8_epi16,
#endif
                             od_tx16_mm256_kernel kernel8_epi32) {
#if CONFIG_RECT_TX_EXT
  if (rows <= 4) {
    __m128i s0;
    __m128i s1;
    __m128i s2;
    __m128i s3;
    __m128i s4;
    __m128i s5;
    __m128i s6;
    __m128i s7;
    __m128i s8;
    __m128i s9;
    __m128i sa;
    __m128i sb;
    __m128i sc;
    __m128i sd;
    __m128i se;
    __m128i sf;
    od_load_buffer_4x4_epi32(&s0, &s1, &s8, &s9, in);
    od_load_buffer_4x4_epi32(&s2, &s3, &sa, &sb, in + 16);
    od_load_buffer_4x4_epi32(&s4, &s5, &sc, &sd, in + 32);
    od_load_buffer_4x4_epi32(&s6, &s7, &se, &sf, in + 48);
    /*TODO(any): Merge this transpose with coefficient scanning.*/
    od_transpose_pack8x4(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
    od_transpose_pack8x4(&s8, &s9, &sa, &sb, &sc, &sd, &se, &sf);
    kernel8_epi16(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8, &s9, &sa, &sb,
                  &sc, &sd, &se, &sf);
    od_transpose4x8(&s0, s8, &s4, sc, &s2, sa, &s6, se);
    od_transpose4x8(&s1, s9, &s5, sd, &s3, sb, &s7, sf);
    od_store_buffer_4x4_epi16(out, s0, s1);
    od_store_buffer_4x4_epi16(out + 16, s4, s5);
    od_store_buffer_4x4_epi16(out + 32, s2, s3);
    od_store_buffer_4x4_epi16(out + 48, s6, s7);
    return;
  }
#endif  // CONFIG_RECT_TX_EXT
  {
    int r;
    /* 8 or more rows requires 32-bit precision.
       TODO(any): If the column TX is IDTX, then we can still use 16 bits. */
    for (r = 0; r < rows; r += 8) {
      __m256i ss0;
      __m256i ss1;
      __m256i ss2;
      __m256i ss3;
      __m256i ss4;
      __m256i ss5;
      __m256i ss6;
      __m256i ss7;
      __m256i ss8;
      __m256i ss9;
      __m256i ssa;
      __m256i ssb;
      __m256i ssc;
      __m256i ssd;
      __m256i sse;
      __m256i ssf;
      od_load_buffer_8x4_epi32(&ss0, &ss8, &ss1, &ss9, in + r * 16);
      od_load_buffer_8x4_epi32(&ss2, &ssa, &ss3, &ssb, in + r * 16 + 32);
      od_load_buffer_8x4_epi32(&ss4, &ssc, &ss5, &ssd, in + r * 16 + 64);
      od_load_buffer_8x4_epi32(&ss6, &sse, &ss7, &ssf, in + r * 16 + 96);
      od_transpose8x8_epi32(&ss0, &ss1, &ss2, &ss3, &ss4, &ss5, &ss6, &ss7);
      od_transpose8x8_epi32(&ss8, &ss9, &ssa, &ssb, &ssc, &ssd, &sse, &ssf);
      kernel8_epi32(&ss0, &ss1, &ss2, &ss3, &ss4, &ss5, &ss6, &ss7, &ss8, &ss9,
                    &ssa, &ssb, &ssc, &ssd, &sse, &ssf);
      od_transpose_pack8x16_epi32(&ss0, &ss8, &ss4, &ssc, &ss2, &ssa, &ss6,
                                  &sse, ss1, ss9, ss5, ssd, ss3, ssb, ss7, ssf);
      od_store_buffer_2x16_epi16(out + r * 16, ss0, ss8);
      od_store_buffer_2x16_epi16(out + r * 16 + 32, ss4, ssc);
      od_store_buffer_2x16_epi16(out + r * 16 + 64, ss2, ssa);
      od_store_buffer_2x16_epi16(out + r * 16 + 96, ss6, sse);
    }
  }
}

static void od_col_tx16_add_hbd_avx2(unsigned char *output_pixels,
                                     int output_stride, int cols,
                                     const int16_t *in, int bd,
                                     od_tx16_kernel8_epi16 kernel8_epi16,
                                     od_tx16_mm256_kernel kernel16_epi16) {
  __m128i s0;
  __m128i s1;
  __m128i s2;
  __m128i s3;
  __m128i s4;
  __m128i s5;
  __m128i s6;
  __m128i s7;
  __m128i s8;
  __m128i s9;
  __m128i sa;
  __m128i sb;
  __m128i sc;
  __m128i sd;
  __m128i se;
  __m128i sf;
#if CONFIG_RECT_TX_EXT
  if (cols <= 4) {
    od_load_buffer_4x4_epi16(&s0, &s1, &s2, &s3, in);
    od_load_buffer_4x4_epi16(&s4, &s5, &s6, &s7, in + 16);
    od_load_buffer_4x4_epi16(&s8, &s9, &sa, &sb, in + 32);
    od_load_buffer_4x4_epi16(&sc, &sd, &se, &sf, in + 48);
    kernel8_epi16(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8, &s9, &sa, &sb,
                  &sc, &sd, &se, &sf);
    od_add_store_buffer_hbd_4x4_epi16(output_pixels, output_stride, s0, s8, s4,
                                      sc, bd);
    od_add_store_buffer_hbd_4x4_epi16(output_pixels + 4 * output_stride,
                                      output_stride, s2, sa, s6, se, bd);
    od_add_store_buffer_hbd_4x4_epi16(output_pixels + 8 * output_stride,
                                      output_stride, s1, s9, s5, sd, bd);
    od_add_store_buffer_hbd_4x4_epi16(output_pixels + 12 * output_stride,
                                      output_stride, s3, sb, s7, sf, bd);
    return;
  }
#endif  // CONFIG_RECT_TX_EXT
  if (cols <= 8) {
    od_load_buffer_8x4_epi16(&s0, &s1, &s2, &s3, in, cols);
    od_load_buffer_8x4_epi16(&s4, &s5, &s6, &s7, in + 32, cols);
    od_load_buffer_8x4_epi16(&s8, &s9, &sa, &sb, in + 64, cols);
    od_load_buffer_8x4_epi16(&sc, &sd, &se, &sf, in + 96, cols);
    kernel8_epi16(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7, &s8, &s9, &sa, &sb,
                  &sc, &sd, &se, &sf);
    od_add_store_buffer_hbd_8x4_epi16(output_pixels, output_stride, s0, s8, s4,
                                      sc, bd);
    od_add_store_buffer_hbd_8x4_epi16(output_pixels + 4 * output_stride,
                                      output_stride, s2, sa, s6, se, bd);
    od_add_store_buffer_hbd_8x4_epi16(output_pixels + 8 * output_stride,
                                      output_stride, s1, s9, s5, sd, bd);
    od_add_store_buffer_hbd_8x4_epi16(output_pixels + 12 * output_stride,
                                      output_stride, s3, sb, s7, sf, bd);
  } else {
    __m256i ss0;
    __m256i ss1;
    __m256i ss2;
    __m256i ss3;
    __m256i ss4;
    __m256i ss5;
    __m256i ss6;
    __m256i ss7;
    __m256i ss8;
    __m256i ss9;
    __m256i ssa;
    __m256i ssb;
    __m256i ssc;
    __m256i ssd;
    __m256i sse;
    __m256i ssf;
    int c;
    for (c = 0; c < cols; c += 16) {
      od_load_buffer_16x4_epi16(&ss0, &ss1, &ss2, &ss3, in + c, cols);
      od_load_buffer_16x4_epi16(&ss4, &ss5, &ss6, &ss7, in + 4 * cols + c,
                                cols);
      od_load_buffer_16x4_epi16(&ss8, &ss9, &ssa, &ssb, in + 8 * cols + c,
                                cols);
      od_load_buffer_16x4_epi16(&ssc, &ssd, &sse, &ssf, in + 12 * cols + c,
                                cols);
      kernel16_epi16(&ss0, &ss1, &ss2, &ss3, &ss4, &ss5, &ss6, &ss7, &ss8, &ss9,
                     &ssa, &ssb, &ssc, &ssd, &sse, &ssf);
      od_add_store_buffer_hbd_16x4_epi16(output_pixels, output_stride, ss0, ss8,
                                         ss4, ssc, bd);
      od_add_store_buffer_hbd_16x4_epi16(output_pixels + 4 * output_stride,
                                         output_stride, ss2, ssa, ss6, sse, bd);
      od_add_store_buffer_hbd_16x4_epi16(output_pixels + 8 * output_stride,
                                         output_stride, ss1, ss9, ss5, ssd, bd);
      od_add_store_buffer_hbd_16x4_epi16(output_pixels + 12 * output_stride,
                                         output_stride, ss3, ssb, ss7, ssf, bd);
    }
  }
}

static void od_row_idct16_avx2(int16_t *out, int rows, const tran_low_t *in) {
  od_row_tx16_avx2(out, rows, in,
#if CONFIG_RECT_TX_EXT
                   od_idct16_kernel8_epi16,
#endif
                   od_idct16_kernel8_epi32);
}

static void od_col_idct16_add_hbd_avx2(unsigned char *output_pixels,
                                       int output_stride, int cols,
                                       const int16_t *in, int bd) {
  od_col_tx16_add_hbd_avx2(output_pixels, output_stride, cols, in, bd,
                           od_idct16_kernel8_epi16, od_idct16_kernel16_epi16);
}

typedef void (*daala_row_itx)(int16_t *out, int rows, const tran_low_t *in);
typedef void (*daala_col_itx_add)(unsigned char *output_pixels,
                                  int output_stride, int cols,
                                  const int16_t *in, int bd);

static const daala_row_itx TX_ROW_MAP[TX_SIZES][TX_TYPES] = {
  // 4-point transforms
  { od_row_idct4_avx2, od_row_idst4_avx2, od_row_flip_idst4_avx2,
    od_row_iidtx4_avx2 },
  // 8-point transforms
  { od_row_idct8_avx2, od_row_idst8_avx2, od_row_flip_idst8_avx2,
    od_row_iidtx8_avx2 },
  // 16-point transforms
  { od_row_idct16_avx2, NULL, NULL, NULL },
  // 32-point transforms
  { NULL, NULL, NULL, NULL },
#if CONFIG_TX64X64
  // 64-point transforms
  { NULL, NULL, NULL, NULL },
#endif
};

static const daala_col_itx_add TX_COL_MAP[2][TX_SIZES][TX_TYPES] = {
  // Low bit depth output
  {
      // 4-point transforms
      { NULL, NULL, NULL, NULL },
      // 8-point transforms
      { NULL, NULL, NULL, NULL },
      // 16-point transforms
      { NULL, NULL, NULL, NULL },
      // 32-point transforms
      { NULL, NULL, NULL, NULL },
#if CONFIG_TX64X64
      // 64-point transforms
      { NULL, NULL, NULL, NULL },
#endif
  },
  // High bit depth output
  {
      // 4-point transforms
      { od_col_idct4_add_hbd_avx2, od_col_idst4_add_hbd_avx2,
        od_col_flip_idst4_add_hbd_avx2, od_col_iidtx4_add_hbd_avx2 },
      // 8-point transforms
      { od_col_idct8_add_hbd_avx2, od_col_idst8_add_hbd_avx2,
        od_col_flip_idst8_add_hbd_avx2, od_col_iidtx8_add_hbd_avx2 },
      // 16-point transforms
      { od_col_idct16_add_hbd_avx2, NULL, NULL, NULL },
      // 32-point transforms
      { NULL, NULL, NULL, NULL },
#if CONFIG_TX64X64
      // 64-point transforms
      { NULL, NULL, NULL, NULL },
#endif
  }
};

/* Define this to verify the SIMD against the C versions of the transforms.
   This is intended to be replaced by real unit tests in the future. */
#undef DAALA_TX_VERIFY_SIMD

void daala_inv_txfm_add_avx2(const tran_low_t *input_coeffs,
                             void *output_pixels, int output_stride,
                             TxfmParam *txfm_param) {
  const TX_SIZE tx_size = txfm_param->tx_size;
  const TX_TYPE tx_type = txfm_param->tx_type;
  assert(tx_size <= TX_SIZES_ALL);
  assert(tx_type <= TX_TYPES);

  if (txfm_param->lossless) {
    daala_inv_txfm_add_c(input_coeffs, output_pixels, output_stride,
                         txfm_param);
  } else {
    // General TX case
    assert(sizeof(tran_low_t) == sizeof(od_coeff));
    assert(sizeof(tran_low_t) >= 4);

    // Hook into existing map translation infrastructure to select
    // appropriate TX functions
    const TX_SIZE col_idx = txsize_vert_map[tx_size];
    const TX_SIZE row_idx = txsize_horz_map[tx_size];
    assert(col_idx <= TX_SIZES);
    assert(row_idx <= TX_SIZES);
    assert(vtx_tab[tx_type] <= (int)TX_TYPES_1D);
    assert(htx_tab[tx_type] <= (int)TX_TYPES_1D);
    daala_row_itx row_tx = TX_ROW_MAP[row_idx][htx_tab[tx_type]];
    daala_col_itx_add col_tx =
        TX_COL_MAP[txfm_param->is_hbd][col_idx][vtx_tab[tx_type]];
    int16_t tmpsq[MAX_TX_SQUARE];

    if (row_tx == NULL || col_tx == NULL) {
      daala_inv_txfm_add_c(input_coeffs, output_pixels, output_stride,
                           txfm_param);
    } else {
      const int cols = tx_size_wide[tx_size];
      const int rows = tx_size_high[tx_size];
#if defined(DAALA_TX_VERIFY_SIMD)
      unsigned char out_check_buf8[MAX_TX_SQUARE];
      int16_t out_check_buf16[MAX_TX_SQUARE];
      unsigned char *out_check_buf;
      {
        if (txfm_param->is_hbd) {
          uint16_t *output_pixels16;
          int r;
          output_pixels16 = CONVERT_TO_SHORTPTR(output_pixels);
          for (r = 0; r < rows; r++) {
            memcpy(out_check_buf16 + r * cols,
                   output_pixels16 + r * output_stride,
                   cols * sizeof(*out_check_buf16));
          }
          out_check_buf = CONVERT_TO_BYTEPTR(out_check_buf16);
        } else {
          unsigned char *output_pixels8;
          int r;
          output_pixels8 = (unsigned char *)output_pixels;
          for (r = 0; r < rows; r++) {
            memcpy(out_check_buf8 + r * cols,
                   output_pixels8 + r * output_stride,
                   cols * sizeof(*out_check_buf8));
          }
          out_check_buf = out_check_buf8;
        }
      }
      daala_inv_txfm_add_c(input_coeffs, out_check_buf, cols, txfm_param);
#endif
      // Inverse-transform rows
      row_tx(tmpsq, rows, input_coeffs);
      // Inverse-transform columns and sum with destination
      col_tx(output_pixels, output_stride, cols, tmpsq, txfm_param->bd);
#if defined(DAALA_TX_VERIFY_SIMD)
      {
        if (txfm_param->is_hbd) {
          uint16_t *output_pixels16;
          int r;
          output_pixels16 = CONVERT_TO_SHORTPTR(output_pixels);
          for (r = 0; r < rows; r++) {
            if (memcmp(out_check_buf16 + r * cols,
                       output_pixels16 + r * output_stride,
                       cols * sizeof(*out_check_buf16))) {
              fprintf(stderr, "%s(%i): Inverse %ix%i %i_%i TX SIMD mismatch.\n",
                      __FILE__, __LINE__, rows, cols, vtx_tab[tx_type],
                      htx_tab[tx_type]);
              assert(0);
              exit(EXIT_FAILURE);
            }
          }
        } else {
          unsigned char *output_pixels8;
          int r;
          output_pixels8 = (unsigned char *)output_pixels;
          for (r = 0; r < rows; r++) {
            if (memcmp(out_check_buf8 + r * cols,
                       output_pixels8 + r * output_stride,
                       cols * sizeof(*out_check_buf8))) {
              fprintf(stderr, "%s(%i): Inverse %ix%i %i_%i TX SIMD mismatch.\n",
                      __FILE__, __LINE__, rows, cols, vtx_tab[tx_type],
                      htx_tab[tx_type]);
              assert(0);
              exit(EXIT_FAILURE);
            }
          }
        }
      }
#endif
    }
  }
}

#endif
