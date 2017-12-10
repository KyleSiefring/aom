#include <tmmintrin.h>
#include <immintrin.h>
#include "./av1_rtcd.h"
#include "./aom_config.h"
#include "./aom_dsp_rtcd.h"
#include "av1/common/daala_tx.h"
#include "av1/encoder/daala_fwd_txfm.h"
#include "av1/common/x86/daala_tx_avx2.h"

#if CONFIG_DAALA_TX

static INLINE void od_8x4_cvtepi16_epi32(__m128i *q0, __m128i *q1, __m128i *q2, __m128i *q3, __m128i *extra) {
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

/* Loads a 4x4 buffer of 16-bit values, shifts them up to full bit depth, and
   puts them into four SSE registers. */
static INLINE void od_load_buffer_hbd_4x4_epi16(__m128i *q0, __m128i *q1, __m128i *q2, __m128i *q3,
                                            const int16_t *in, int input_stride, int bd) {
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
static INLINE void od_load_buffer_hbd_8x4_epi16(__m128i *q0, __m128i *q1, __m128i *q2, __m128i *q3,
                                            const int16_t *in, int input_stride, int bd) {
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

typedef void (*od_tx4_kernel8_epi16)(__m128i *q0, __m128i *q2, __m128i *q1,
                                     __m128i *q3);

static void od_col_tx4_avx2(int16_t *out, int cols, const int16_t *in, int input_stride,
                            int bd, od_tx4_kernel8_epi16 kernel8) {
  __m128i q0;
  __m128i q1;
  __m128i q2;
  __m128i q3;
  if (cols <= 4) {
    od_load_buffer_hbd_4x4_epi16(&q0, &q1, &q2, &q3, in, input_stride, bd);
    kernel8(&q0, &q1, &q2, &q3);
    od_transpose4x4(&q0, q2, &q1, q3);
    od_store_buffer_4x4_epi16(out, q0, q1);
  } else {
    /* Higher col count unimplemented.*/
    assert(cols <= 8);
    od_load_buffer_hbd_8x4_epi16(&q0, &q1, &q2, &q3, in, input_stride, bd);
    kernel8(&q0, &q1, &q2, &q3);
    od_transpose8x4(&q0, &q2, &q1, &q3);
    od_store_buffer_4x8_epi16(out, q0, q2, q1, q3);
  }
}

static void od_row_tx4_avx2(tran_low_t *output_coeffs,
                            int rows, const int16_t *in,
                            od_tx4_kernel8_epi16 kernel8) {
  __m128i q0;
  __m128i q1;
  __m128i q2;
  __m128i q3;
  if (rows <= 4) {
    od_load_buffer_4x4_epi16(&q0, &q1, &q2, &q3, in);
    kernel8(&q0, &q1, &q2, &q3);
    /*TODO(any): Merge this transpose with coefficient scanning.*/
    od_transpose_unpack4x4(&q0, &q2, &q1, &q3);
    od_store_buffer_4x4_epi32(output_coeffs, q0, q2, q1, q3);
  } else {
    __m128i ext[4];
    /* Higher row count unimplemented.*/
    assert(rows <= 8);
    od_load_buffer_8x4_epi16(&q0, &q1, &q2, &q3, in, rows);
    kernel8(&q0, &q1, &q2, &q3);
    /*TODO(any): Merge this transpose with coefficient scanning.*/
    od_transpose8x4(&q0, &q2, &q1, &q3);
    od_8x4_cvtepi16_epi32(&q0, &q1, &q2, &q3, ext);
    od_store_buffer_8x4_epi32(output_coeffs, q0, ext[0], q2, ext[2], q1, ext[1], q3, ext[3]);
  }
}

static void od_col_fdct4_avx2(int16_t *out, int cols, const int16_t *in, int input_stride, int bd) {
  od_col_tx4_avx2(out, cols, in, input_stride, bd, od_fdct4_kernel8_epi16);
}

static void od_row_fdct4_avx2(tran_low_t *output_coeffs, int rows, const int16_t *in) {
  od_row_tx4_avx2(output_coeffs, rows, in, od_fdct4_kernel8_epi16);
}

typedef void (*od_tx8_kernel8_epi16)(__m128i *r0, __m128i *r4, __m128i *r2,
                                     __m128i *r6, __m128i *r1, __m128i *r5,
                                     __m128i *r3, __m128i *r7);

static void od_col_tx8_avx2(int16_t *out, int cols, const int16_t *in, int input_stride,
                            int bd, od_tx8_kernel8_epi16 kernel8) {
  __m128i r0;
  __m128i r1;
  __m128i r2;
  __m128i r3;
  __m128i r4;
  __m128i r5;
  __m128i r6;
  __m128i r7;
  if (cols <= 4) {
    od_load_buffer_hbd_4x4_epi16(&r0, &r1, &r2, &r3, in, input_stride, bd);
    od_load_buffer_hbd_4x4_epi16(&r4, &r5, &r6, &r7, in + 4*input_stride, input_stride, bd);
    kernel8(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    od_transpose4x8(&r0, r4, &r2, r6, &r1, r5, &r3, r7);
    od_store_buffer_4x8_epi16(out, r0, r2, r1, r3);
  } else {
    /* Higher col count unimplemented.*/
    assert(cols <= 8);
    od_load_buffer_hbd_8x4_epi16(&r0, &r1, &r2, &r3, in, input_stride, bd);
    od_load_buffer_hbd_8x4_epi16(&r4, &r5, &r6, &r7, in + 4 * input_stride, input_stride, bd);
    kernel8(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    od_transpose8x8_epi16(&r0, &r4, &r2, &r6, &r1, &r5, &r3, &r7);
    od_store_buffer_4x8_epi16(out, r0, r4, r2, r6);
    od_store_buffer_4x8_epi16(out + 32, r1, r5, r3, r7);
  }
}

static void od_row_tx8_avx2(tran_low_t *output_coeffs,
                            int rows, const int16_t *in,
                            od_tx8_kernel8_epi16 kernel8) {
  __m128i r0;
  __m128i r1;
  __m128i r2;
  __m128i r3;
  __m128i r4;
  __m128i r5;
  __m128i r6;
  __m128i r7;
  if (rows <= 4) {
    od_load_buffer_4x4_epi16(&r0, &r1, &r2, &r3, in);
    od_load_buffer_4x4_epi16(&r4, &r5, &r6, &r7, in + 16);
    kernel8(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    /*TODO(any): Merge this transpose with coefficient scanning.*/
    od_transpose_unpack4x4(&r0, &r4, &r2, &r6);
    od_transpose_unpack4x4(&r1, &r5, &r3, &r7);
    od_store_buffer_4x4_epi32(output_coeffs, r0, r1, r4, r5);
    od_store_buffer_4x4_epi32(output_coeffs + 16, r2, r3, r6, r7);
  } else {
    /* Higher row count unimplemented.*/
    assert(rows <= 8);
    __m128i ext[8];
    od_load_buffer_8x4_epi16(&r0, &r1, &r2, &r3, in, rows);
    od_load_buffer_8x4_epi16(&r4, &r5, &r6, &r7, in + 32, rows);
    kernel8(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    /*TODO(any): Merge this transpose with coefficient scanning.*/
    od_transpose8x8_epi16(&r0, &r4, &r2, &r6, &r1, &r5, &r3, &r7);
    od_8x4_cvtepi16_epi32(&r0, &r1, &r2, &r3, ext);
    od_8x4_cvtepi16_epi32(&r4, &r5, &r6, &r7, ext + 4);
    od_store_buffer_8x4_epi32(output_coeffs, r0, ext[0], r4, ext[4], r2, ext[2], r6, ext[6]);
    od_store_buffer_8x4_epi32(output_coeffs + 32, r1, ext[1], r5, ext[5], r3, ext[3], r7, ext[7]);
  }
}

static void od_row_fdct8_avx2(tran_low_t *output_coeffs, int rows, const int16_t *in) {
  od_row_tx8_avx2(output_coeffs, rows, in, od_fdct8_kernel8_epi16);
}

static void od_col_fdct8_avx2(int16_t *out, int cols, const int16_t *in, int input_stride, int bd) {
  od_col_tx8_avx2(out, cols, in, input_stride, bd, od_fdct8_kernel8_epi16);
}

typedef void (*daala_col_ftx)(int16_t *out, int cols, const int16_t *in, int input_stride, int bd);
typedef void (*daala_row_ftx)(tran_low_t *output_coeffs, int rows, const int16_t *in);

static const daala_col_ftx TX_COL_MAP[TX_SIZES][TX_TYPES] = {
  // 4-point transforms
  { od_col_fdct4_avx2, NULL, NULL, NULL },
  // 8-point transforms
  { od_col_fdct8_avx2, NULL, NULL, NULL },
  // 16-point transforms
  { NULL, NULL, NULL, NULL },
  // 32-point transforms
  { NULL, NULL, NULL, NULL },
#if CONFIG_TX64X64
  // 64-point transforms
  { NULL, NULL, NULL, NULL },
#endif
};

static const daala_row_ftx TX_ROW_MAP[TX_SIZES][TX_TYPES] = {
  // 4-point transforms
  { od_row_fdct4_avx2, NULL, NULL, NULL },
  // 8-point transforms
  { od_row_fdct8_avx2, NULL, NULL, NULL },
  // 16-point transforms
  { NULL, NULL, NULL, NULL },
  // 32-point transforms
  { NULL, NULL, NULL, NULL },
#if CONFIG_TX64X64
  // 64-point transforms
  { NULL, NULL, NULL, NULL },
#endif
};

/* Define this to verify the SIMD against the C versions of the transforms.
   This is intended to be replaced by real unit tests in the future. */
// TODO Remember
#define DAALA_TX_VERIFY_SIMD

void daala_fwd_txfm_avx2(const int16_t *input_pixels, tran_low_t *output_coeffs,
                         int input_stride, TxfmParam *txfm_param) {
  //const TX_SIZE tx_size = txfm_param->tx_size;
  TX_SIZE tx_size = txfm_param->tx_size;
  const TX_TYPE tx_type = txfm_param->tx_type;
  assert(tx_size <= TX_SIZES_ALL);
  assert(tx_type <= TX_TYPES);

  if (txfm_param->lossless) {
    daala_fwd_txfm_c(input_pixels, output_coeffs, input_stride, txfm_param);
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
    daala_col_ftx col_tx = TX_COL_MAP[col_idx][vtx_tab[tx_type]];
    daala_row_ftx row_tx = TX_ROW_MAP[row_idx][htx_tab[tx_type]];

    int16_t tmpsq[MAX_TX_SQUARE];
    if (col_tx == NULL || row_tx == NULL) {
      daala_fwd_txfm_c(input_pixels, output_coeffs, input_stride, txfm_param);
    } else {
      const int cols = tx_size_wide[tx_size];
      const int rows = tx_size_high[tx_size];
      // Forward-transform columns
      col_tx(tmpsq, cols, input_pixels, input_stride, txfm_param->bd);
      // Forward-transform columns and sum with destination
      row_tx(output_coeffs, rows, tmpsq);
#if defined(DAALA_TX_VERIFY_SIMD)
      tran_low_t out_check_buf[MAX_TX_SQUARE];
      daala_fwd_txfm_c(input_pixels, out_check_buf, input_stride, txfm_param);
      {
        int r;
        for (r = 0; r < rows; r++) {
          if (memcmp(out_check_buf + r * cols,
                     output_coeffs + r * cols,
                     cols * sizeof(*out_check_buf))) {
            fprintf(stderr, "%s(%i): Forward %ix%i %i_%i TX SIMD mismatch.\n",
                    __FILE__, __LINE__, rows, cols, vtx_tab[tx_type],
                    htx_tab[tx_type]);
            assert(0);
            exit(EXIT_FAILURE);
          }
        }
      }
#endif
    }
  }
}

#endif
