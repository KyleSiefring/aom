/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <immintrin.h>

#include "./aom_dsp_rtcd.h"
#include "aom_dsp/aom_convolve.h"
#include "aom_dsp/x86/convolve_avx2.h"
#include "aom_dsp/x86/convolve_common_intrin.h"
#include "aom_dsp/x86/convolve_sse4_1.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/aom_filter.h"
#include "av1/common/convolve.h"

void av1_jnt_convolve_x_avx2(const uint8_t *src, int src_stride,
                             const uint8_t *dst0, int dst_stride0, int w, int h,
                             InterpFilterParams *filter_params_x,
                             InterpFilterParams *filter_params_y,
                             const int subpel_x_q4, const int subpel_y_q4,
                             ConvolveParams *conv_params) {
  CONV_BUF_TYPE *dst = conv_params->dst;
  int dst_stride = conv_params->dst_stride;
  int i, j;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const uint8_t *const src_ptr = src - fo_horiz;
  const int bits = FILTER_BITS - conv_params->round_1;
  const int w0 = conv_params->fwd_offset;
  const int w1 = conv_params->bck_offset;
  const __m256i wt0 = _mm256_set1_epi32(w0);
  const __m256i wt1 = _mm256_set1_epi32(w1);
  const int do_average = conv_params->do_average;
  __m256i filt[4], coeffs[4];

  assert(bits >= 0);
  assert(conv_params->round_0 > 0);

  filt[0] = _mm256_load_si256((__m256i const *)filt1_global_avx2);
  filt[1] = _mm256_load_si256((__m256i const *)filt2_global_avx2);
  filt[2] = _mm256_load_si256((__m256i const *)filt3_global_avx2);
  filt[3] = _mm256_load_si256((__m256i const *)filt4_global_avx2);

  prepare_coeffs_lowbd(filter_params_x, subpel_x_q4, coeffs);

  const __m256i round_const =
      _mm256_set1_epi16((1 << (conv_params->round_0 - 1)) >> 1);
  const __m128i round_shift = _mm_cvtsi32_si128(conv_params->round_0 - 1);

  (void)filter_params_y;
  (void)subpel_y_q4;
  (void)dst0;
  (void)dst_stride0;

  for (i = 0; i < h; i += 2) {
    for (j = 0; j < w; j += 8) {
      const __m256i data = _mm256_permute2x128_si256(
          _mm256_castsi128_si256(
              _mm_loadu_si128((__m128i *)(&src_ptr[i * src_stride + j]))),
          _mm256_castsi128_si256(_mm_loadu_si128(
              (__m128i *)(&src_ptr[i * src_stride + j + src_stride]))),
          0x20);

      __m256i res = convolve_lowbd_x(data, coeffs, filt);

      res = _mm256_sra_epi16(_mm256_add_epi16(res, round_const), round_shift);

      const __m256i res_lo_round =
          _mm256_cvtepi16_epi32(_mm256_castsi256_si128(res));
      const __m256i res_hi_round =
          _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res, 1));

      const __m256i res_lo_shift = _mm256_slli_epi32(res_lo_round, bits);
      const __m256i res_hi_shift = _mm256_slli_epi32(res_hi_round, bits);

      // Accumulate values into the destination buffer
      if (conv_params->use_jnt_comp_avg) {
        mult_add_store_aligned_256(&dst[i * dst_stride + j], &res_lo_shift,
                                   &wt0, &wt1, do_average);
        mult_add_store_aligned_256(&dst[i * dst_stride + j + dst_stride],
                                   &res_hi_shift, &wt0, &wt1, do_average);
      } else {
        add_store_aligned_256(&dst[i * dst_stride + j], &res_lo_shift,
                              do_average);
        add_store_aligned_256(&dst[i * dst_stride + j + dst_stride],
                              &res_hi_shift, do_average);
      }
    }
  }
}

void av1_jnt_convolve_y_avx2(const uint8_t *src, int src_stride,
                             const uint8_t *dst0, int dst_stride0, int w, int h,
                             InterpFilterParams *filter_params_x,
                             InterpFilterParams *filter_params_y,
                             const int subpel_x_q4, const int subpel_y_q4,
                             ConvolveParams *conv_params) {
  CONV_BUF_TYPE *dst = conv_params->dst;
  int dst_stride = conv_params->dst_stride;
  int i, j;
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const uint8_t *const src_ptr = src - fo_vert * src_stride;
  // +1 to compensate for dividing the filter coeffs by 2
  const int left_shift = FILTER_BITS - conv_params->round_0 + 1;
  const __m256i round_const =
      _mm256_set1_epi32((1 << conv_params->round_1) >> 1);
  const __m128i round_shift = _mm_cvtsi32_si128(conv_params->round_1);
  const int w0 = conv_params->fwd_offset;
  const int w1 = conv_params->bck_offset;
  const __m256i wt0 = _mm256_set1_epi32(w0);
  const __m256i wt1 = _mm256_set1_epi32(w1);
  const int do_average = conv_params->do_average;
  __m256i coeffs[4], s[8];

  assert((FILTER_BITS - conv_params->round_0) >= 0);

  prepare_coeffs_lowbd(filter_params_y, subpel_y_q4, coeffs);

  (void)conv_params;
  (void)filter_params_x;
  (void)subpel_x_q4;
  (void)dst0;
  (void)dst_stride0;

  for (j = 0; j < w; j += 16) {
    const uint8_t *data = &src_ptr[j];
    __m256i src6;

    // Load lines a and b. Line a to lower 128, line b to upper 128
    const __m256i src_01a = _mm256_permute2x128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 0 * src_stride))),
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 1 * src_stride))),
        0x20);

    const __m256i src_12a = _mm256_permute2x128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 1 * src_stride))),
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 2 * src_stride))),
        0x20);

    const __m256i src_23a = _mm256_permute2x128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 2 * src_stride))),
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 3 * src_stride))),
        0x20);

    const __m256i src_34a = _mm256_permute2x128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 3 * src_stride))),
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 4 * src_stride))),
        0x20);

    const __m256i src_45a = _mm256_permute2x128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 4 * src_stride))),
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 5 * src_stride))),
        0x20);

    src6 = _mm256_castsi128_si256(
        _mm_loadu_si128((__m128i *)(data + 6 * src_stride)));
    const __m256i src_56a = _mm256_permute2x128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(data + 5 * src_stride))),
        src6, 0x20);

    s[0] = _mm256_unpacklo_epi8(src_01a, src_12a);
    s[1] = _mm256_unpacklo_epi8(src_23a, src_34a);
    s[2] = _mm256_unpacklo_epi8(src_45a, src_56a);

    s[4] = _mm256_unpackhi_epi8(src_01a, src_12a);
    s[5] = _mm256_unpackhi_epi8(src_23a, src_34a);
    s[6] = _mm256_unpackhi_epi8(src_45a, src_56a);

    for (i = 0; i < h; i += 2) {
      data = &src_ptr[i * src_stride + j];
      const __m256i src_67a = _mm256_permute2x128_si256(
          src6,
          _mm256_castsi128_si256(
              _mm_loadu_si128((__m128i *)(data + 7 * src_stride))),
          0x20);

      src6 = _mm256_castsi128_si256(
          _mm_loadu_si128((__m128i *)(data + 8 * src_stride)));
      const __m256i src_78a = _mm256_permute2x128_si256(
          _mm256_castsi128_si256(
              _mm_loadu_si128((__m128i *)(data + 7 * src_stride))),
          src6, 0x20);

      s[3] = _mm256_unpacklo_epi8(src_67a, src_78a);
      s[7] = _mm256_unpackhi_epi8(src_67a, src_78a);

      const __m256i res_lo = convolve_lowbd(s, coeffs);

      const __m256i res_lo_0_32b =
          _mm256_cvtepi16_epi32(_mm256_castsi256_si128(res_lo));
      const __m256i res_lo_0_shift =
          _mm256_slli_epi32(res_lo_0_32b, left_shift);
      const __m256i res_lo_0_round = _mm256_sra_epi32(
          _mm256_add_epi32(res_lo_0_shift, round_const), round_shift);

      const __m256i res_lo_1_32b =
          _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_lo, 1));
      const __m256i res_lo_1_shift =
          _mm256_slli_epi32(res_lo_1_32b, left_shift);
      const __m256i res_lo_1_round = _mm256_sra_epi32(
          _mm256_add_epi32(res_lo_1_shift, round_const), round_shift);

      // Accumulate values into the destination buffer
      if (conv_params->use_jnt_comp_avg) {
        mult_add_store_aligned_256(&dst[i * dst_stride + j], &res_lo_0_round,
                                   &wt0, &wt1, do_average);
        mult_add_store_aligned_256(&dst[i * dst_stride + j + dst_stride],
                                   &res_lo_1_round, &wt0, &wt1, do_average);
      } else {
        add_store_aligned_256(&dst[i * dst_stride + j], &res_lo_0_round,
                              do_average);
        add_store_aligned_256(&dst[i * dst_stride + j + dst_stride],
                              &res_lo_1_round, do_average);
      }

      if (w - j > 8) {
        const __m256i res_hi = convolve_lowbd(s + 4, coeffs);

        const __m256i res_hi_0_32b =
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(res_hi));
        const __m256i res_hi_0_shift =
            _mm256_slli_epi32(res_hi_0_32b, left_shift);
        const __m256i res_hi_0_round = _mm256_sra_epi32(
            _mm256_add_epi32(res_hi_0_shift, round_const), round_shift);

        const __m256i res_hi_1_32b =
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256(res_hi, 1));
        const __m256i res_hi_1_shift =
            _mm256_slli_epi32(res_hi_1_32b, left_shift);
        const __m256i res_hi_1_round = _mm256_sra_epi32(
            _mm256_add_epi32(res_hi_1_shift, round_const), round_shift);

        if (conv_params->use_jnt_comp_avg) {
          mult_add_store_aligned_256(&dst[i * dst_stride + j + 8],
                                     &res_hi_0_round, &wt0, &wt1, do_average);
          mult_add_store_aligned_256(&dst[i * dst_stride + j + 8 + dst_stride],
                                     &res_hi_1_round, &wt0, &wt1, do_average);
        } else {
          add_store_aligned_256(&dst[i * dst_stride + j + 8], &res_hi_0_round,
                                do_average);
          add_store_aligned_256(&dst[i * dst_stride + j + 8 + dst_stride],
                                &res_hi_1_round, do_average);
        }
      }
      s[0] = s[1];
      s[1] = s[2];
      s[2] = s[3];

      s[4] = s[5];
      s[5] = s[6];
      s[6] = s[7];
    }
  }
}

void av1_jnt_convolve_2d_avx2(const uint8_t *src, int src_stride, uint8_t *dst0,
                              int dst_stride0, int w, int h,
                              InterpFilterParams *filter_params_x,
                              InterpFilterParams *filter_params_y,
                              const int subpel_x_q4, const int subpel_y_q4,
                              ConvolveParams *conv_params) {
  CONV_BUF_TYPE *dst = conv_params->dst;
  int dst_stride = conv_params->dst_stride;
  const int bd = 8;
  (void)dst0;
  (void)dst_stride0;

  DECLARE_ALIGNED(32, int16_t, im_block[(MAX_SB_SIZE + MAX_FILTER_TAP) * 8]);
  int im_h = h + filter_params_y->taps - 1;
  int im_stride = 8;
  int i, j;
  const int fo_vert = filter_params_y->taps / 2 - 1;
  const int fo_horiz = filter_params_x->taps / 2 - 1;
  const uint8_t *const src_ptr = src - fo_vert * src_stride - fo_horiz;
  const int w0 = conv_params->fwd_offset;
  const int w1 = conv_params->bck_offset;
  const __m256i wt0 = _mm256_set1_epi32(w0);
  const __m256i wt1 = _mm256_set1_epi32(w1);
  const int do_average = conv_params->do_average;
  const __m128i wt0_128 = _mm256_castsi256_si128(wt0);
  const __m128i wt1_128 = _mm256_castsi256_si128(wt1);
  __m256i filt[4], s[8], coeffs_x[4], coeffs_y[4];

  assert(conv_params->round_0 > 0);

  filt[0] = _mm256_load_si256((__m256i const *)filt1_global_avx2);
  filt[1] = _mm256_load_si256((__m256i const *)filt2_global_avx2);
  filt[2] = _mm256_load_si256((__m256i const *)filt3_global_avx2);
  filt[3] = _mm256_load_si256((__m256i const *)filt4_global_avx2);

  prepare_coeffs_lowbd(filter_params_x, subpel_x_q4, coeffs_x);
  prepare_coeffs(filter_params_y, subpel_y_q4, coeffs_y);

  const __m256i round_const_h = _mm256_set1_epi16(
      ((1 << (conv_params->round_0 - 1)) >> 1) + (1 << (bd + FILTER_BITS - 2)));
  const __m128i round_shift_h = _mm_cvtsi32_si128(conv_params->round_0 - 1);

  const __m256i round_const_v = _mm256_set1_epi32(
      ((1 << conv_params->round_1) >> 1) -
      (1 << (bd + 2 * FILTER_BITS - conv_params->round_0 - 1)));
  const __m128i round_shift_v = _mm_cvtsi32_si128(conv_params->round_1);

  for (j = 0; j < w; j += 8) {
    /* Horizontal filter */
    {
      for (i = 0; i < im_h; i += 2) {
        __m256i data = _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)&src_ptr[(i * src_stride) + j]));
        if (i + 1 < im_h)
          data = _mm256_inserti128_si256(
              data,
              _mm_loadu_si128(
                  (__m128i *)&src_ptr[(i * src_stride) + j + src_stride]),
              1);
        __m256i res = convolve_lowbd_x(data, coeffs_x, filt);

        res = _mm256_sra_epi16(_mm256_add_epi16(res, round_const_h),
                               round_shift_h);

        _mm256_store_si256((__m256i *)&im_block[i * im_stride], res);
      }
    }

    /* Vertical filter */
    {
      __m256i s0 = _mm256_loadu_si256((__m256i *)(im_block + 0 * im_stride));
      __m256i s1 = _mm256_loadu_si256((__m256i *)(im_block + 1 * im_stride));
      __m256i s2 = _mm256_loadu_si256((__m256i *)(im_block + 2 * im_stride));
      __m256i s3 = _mm256_loadu_si256((__m256i *)(im_block + 3 * im_stride));
      __m256i s4 = _mm256_loadu_si256((__m256i *)(im_block + 4 * im_stride));
      __m256i s5 = _mm256_loadu_si256((__m256i *)(im_block + 5 * im_stride));

      s[0] = _mm256_unpacklo_epi16(s0, s1);
      s[1] = _mm256_unpacklo_epi16(s2, s3);
      s[2] = _mm256_unpacklo_epi16(s4, s5);

      s[4] = _mm256_unpackhi_epi16(s0, s1);
      s[5] = _mm256_unpackhi_epi16(s2, s3);
      s[6] = _mm256_unpackhi_epi16(s4, s5);

      for (i = 0; i < h; i += 2) {
        const int16_t *data = &im_block[i * im_stride];

        const __m256i s6 =
            _mm256_loadu_si256((__m256i *)(data + 6 * im_stride));
        const __m256i s7 =
            _mm256_loadu_si256((__m256i *)(data + 7 * im_stride));

        s[3] = _mm256_unpacklo_epi16(s6, s7);
        s[7] = _mm256_unpackhi_epi16(s6, s7);

        const __m256i res_a = convolve(s, coeffs_y);
        const __m256i res_a_round = _mm256_sra_epi32(
            _mm256_add_epi32(res_a, round_const_v), round_shift_v);

        if (w - j > 4) {
          const __m256i res_b = convolve(s + 4, coeffs_y);
          const __m256i res_b_round = _mm256_sra_epi32(
              _mm256_add_epi32(res_b, round_const_v), round_shift_v);
          const __m256i res_ax =
              _mm256_permute2x128_si256(res_a_round, res_b_round, 0x20);
          const __m256i res_bx =
              _mm256_permute2x128_si256(res_a_round, res_b_round, 0x31);

          if (conv_params->use_jnt_comp_avg) {
            mult_add_store_aligned_256(&dst[i * dst_stride + j], &res_ax, &wt0,
                                       &wt1, do_average);
            mult_add_store_aligned_256(&dst[i * dst_stride + j + dst_stride],
                                       &res_bx, &wt0, &wt1, do_average);
          } else {
            add_store_aligned_256(&dst[i * dst_stride + j], &res_ax,
                                  do_average);
            add_store_aligned_256(&dst[i * dst_stride + j + dst_stride],
                                  &res_bx, do_average);
          }
        } else {
          const __m128i res_ax = _mm256_castsi256_si128(res_a_round);
          const __m128i res_bx = _mm256_extracti128_si256(res_a_round, 1);

          if (conv_params->use_jnt_comp_avg) {
            mult_add_store(&dst[i * dst_stride + j], &res_ax, &wt0_128,
                           &wt1_128, do_average);
            mult_add_store(&dst[i * dst_stride + j + dst_stride], &res_bx,
                           &wt0_128, &wt1_128, do_average);
          } else {
            add_store(&dst[i * dst_stride + j], &res_ax, do_average);
            add_store(&dst[i * dst_stride + j + dst_stride], &res_bx,
                      do_average);
          }
        }

        s[0] = s[1];
        s[1] = s[2];
        s[2] = s[3];

        s[4] = s[5];
        s[5] = s[6];
        s[6] = s[7];
      }
    }
  }
}

void av1_jnt_convolve_2d_copy_avx2(const uint8_t *src, int src_stride,
                                   uint8_t *dst0, int dst_stride0, int w, int h,
                                   InterpFilterParams *filter_params_x,
                                   InterpFilterParams *filter_params_y,
                                   const int subpel_x_q4, const int subpel_y_q4,
                                   ConvolveParams *conv_params) {
  CONV_BUF_TYPE *dst = conv_params->dst;
  int dst_stride = conv_params->dst_stride;
  (void)filter_params_x;
  (void)filter_params_y;
  (void)subpel_x_q4;
  (void)subpel_y_q4;
  (void)dst0;
  (void)dst_stride0;

  const int bits =
      FILTER_BITS * 2 - conv_params->round_1 - conv_params->round_0;
  const __m128i left_shift = _mm_cvtsi32_si128(bits);
  const int do_average = conv_params->do_average;
  const int w0 = conv_params->fwd_offset;
  const int w1 = conv_params->bck_offset;
  const __m256i wt0 = _mm256_set1_epi32(w0);
  const __m256i wt1 = _mm256_set1_epi32(w1);
  const __m256i zero = _mm256_setzero_si256();
  int i, j;

  if (!(w % 16)) {
    for (i = 0; i < h; i += 1) {
      for (j = 0; j < w; j += 16) {
        const __m256i src_16bit = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i *)(&src[i * src_stride + j])));

        const __m256i res = _mm256_sll_epi16(src_16bit, left_shift);
        const __m256i res_lo =
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(res));
        const __m256i res_hi =
            _mm256_cvtepu16_epi32(_mm256_extracti128_si256(res, 1));

        if (conv_params->use_jnt_comp_avg) {
          mult_add_store_aligned_256(&dst[i * dst_stride + j], &res_lo, &wt0,
                                     &wt1, do_average);
          mult_add_store_aligned_256(&dst[i * dst_stride + j + 8], &res_hi,
                                     &wt0, &wt1, do_average);
        } else {
          add_store_aligned_256(&dst[i * dst_stride + j], &res_lo, do_average);
          add_store_aligned_256(&dst[i * dst_stride + j + 8], &res_hi,
                                do_average);
        }
      }
    }
  } else if (!(w % 4)) {
    for (i = 0; i < h; i += 2) {
      for (j = 0; j < w; j += 8) {
        const __m128i src_row_0 =
            _mm_loadl_epi64((__m128i *)(&src[i * src_stride + j]));
        const __m128i src_row_1 =
            _mm_loadl_epi64((__m128i *)(&src[i * src_stride + j + src_stride]));
        // since not all compilers yet support _mm256_set_m128i()
        const __m256i src_10 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(src_row_0), src_row_1, 1);

        const __m256i src_16bit = _mm256_unpacklo_epi8(src_10, zero);

        const __m256i res = _mm256_sll_epi16(src_16bit, left_shift);

        const __m256i res_lo =
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(res));
        const __m256i res_hi =
            _mm256_cvtepu16_epi32(_mm256_extracti128_si256(res, 1));

        if (conv_params->use_jnt_comp_avg) {
          mult_add_store_aligned_256(&dst[i * dst_stride + j], &res_lo, &wt0,
                                     &wt1, do_average);
          mult_add_store_aligned_256(&dst[i * dst_stride + j + dst_stride],
                                     &res_hi, &wt0, &wt1, do_average);
        } else {
          add_store_aligned_256(&dst[i * dst_stride + j], &res_lo, do_average);
          add_store_aligned_256(&dst[i * dst_stride + j + dst_stride], &res_hi,
                                do_average);
        }
      }
    }
  }
}