/*
 * conv.hpp
 *
 *  Created on: Apr 17, 2016
 *      Author: jpark103
 */

#ifndef _CAFFE_UTIL_CONV_HPP_
#define _CAFFE_UTIL_CONV_HPP_


#include <vector>
// #include <immintrin.h>
#include "SpMP/synk/barrier.hpp"
#include "intrinsic.hpp"


#ifndef PRECISION
#define PRECISION 2
#endif

// #if PRECISION == 2
// #define SIMD_WIDTH 4
// #define SIMDFPTYPE __m256d

// #define _MM_LOAD(a) _mm256_load_pd(a)
// #define _MM_LOADU _mm256_loadu_pd

// #define _MM_STORE(a, v) _mm256_store_pd((a), v)
// #define _MM_STOREU(a, v) _mm256_storeu_pd((a), v)
// #define _MM_MASKSTORE _mm256_maskstore_pd
// #define _MM_STREAM(a, v) _mm256_stream_pd((a), v)
// #define _MM_PREFETCH1(a) _mm_prefetch((char *)(a), _MM_HINT_T0)

// #define _MM_ADD _mm256_add_pd
// #define _MM_MUL _mm256_mul_pd
// #define _MM_ADDSUB _mm256_addsub_pd
// #ifdef __AVX2__
// #define _MM_FMADDSUB _mm256_fmaddsub_pd
// #else
// #define _MM_FMADDSUB(a, b, c) _MM_ADDSUB(_MM_MUL(a, b), c)
// #endif

// #define _MM_SWAP_REAL_IMAG(a) _mm256_permute_pd(a, 0x05)
// #define _MM_MOVELDUP _mm256_movedup_pd
// #define _MM_MOVEHDUP(a) _mm256_permute_pd(a, 0xf)

// #else
// // PRECISION == 1

// #define SIMD_WIDTH 8
// #define SIMDFPTYPE __m256

// #define _MM_LOAD _mm256_load_ps
// #define _MM_LOADU _mm256_loadu_ps

// #define _MM_STORE _mm256_store_ps
// #define _MM_STOREU _mm256_storeu_ps
// #define _MM_MASKSTORE _mm256_maskstore_ps
// #define _MM_STREAM _mm256_stream_ps

// #define _MM_ADD _mm256_add_ps
// #define _MM_MUL _mm256_mul_ps
// #define _MM_ADDSUB _mm256_addsub_ps

// #define _MM_SWAP_REAL_IMAG(a) _mm256_permute_ps(a, 0xb1)
// #define _MM_MOVELDUP _mm256_moveldup_ps
// #define _MM_MOVEHDUP _mm256_movehdup_ps

// #endif // PRECISION == 1

#ifdef __AVX512F__
#ifdef SNIPER
static const int NTILES = 1; // 1 tile
#else
static const int NTILES = 64; // FIXME - hardcoded for 68c KNL
#endif
#endif

static const int OC_BLOCK = 16;

//static const int COL_MAJOR_IC_BLOCK = 8;
//static const int COL_MAJOR_OC_BLOCK = 64;

extern unsigned long long conv_cycles_of_this_batch[1024*16], transpose_cycle, pool_cycle;




static void printv_pd(__m256d v, char *str)
{
  int i;
  __attribute__ ((aligned(64))) double tmp[4];
  printf("%s:", str);
  _mm256_store_pd(tmp, v);
  for(i=0; i < 4; i++)
    printf("[%d]=%g ", i, tmp[i]);
  printf("\n");
}

static void printv_ps(__m256 v, char *str)
{
  int i;
  __attribute__ ((aligned(64))) float tmp[8];
  printf("%s:", str);
  _mm256_store_ps(tmp, v);
  for(i=0; i < 8; i++)
    printf("[%d]=%g ", i, tmp[i]);
  printf("\n");
}


static int get_col_major_ic_block(int nnz, int num_out_channels, int num_in_channels) {
  // # of in-channels to have on average 32 non-zeros per out-channel
  double nnz_per_oc_and_ic = (double)nnz/num_out_channels/num_in_channels;
  int ret = std::max(8, 1 << (int)round(log2(std::max(1., 32/nnz_per_oc_and_ic))));
  ret = std::min(num_in_channels, ret);
  while (num_in_channels%ret != 0) {
    ++ret;
  }
  return ret;
}

extern int flop_cnt;

/**
 * Direct sparse convolution optimized for 3-5 layers of AlexNet, fused with bias term
 *
 * This version involves a lot of unaligned loads
// JSP: AlexNet each group of conv3-5
// Input: 256 x 15 x 15 => 900 B per channel, 225 KB total
// Output: 384 x 13 x 13 => 676 B per channel, 253 KB total
// Weight: 384 x 256 x 3 x 3 => 72B per channel pair, 18 KB per output channel, 27 KB per input channel, 6.8 MB total
//         No matter what we do, there's no reuse on weight across different channels (only reuse is within a channel pair)
// FLOPS: 2 x 384 x 256 x 13 x 13 x 3 x 3 = 299 MFLOPS


 */
template<int WIDTH, int K, bool FUSE_RELU = false, int PAD = (K - 1)/2>
static /*inline*/ void __attribute__((noinline)) sconv_unit_stride(
    // input features
    const float *input,
    // weights
    const int **rowptr_blocked, const int **colidx_blocked, const float **values_blocked,
    int ncolblocks,
    // bias (for the case when bias is fused with convolution)
    const float *bias,
    // output features
    float *output,
    int output_channel_begin, int output_channel_end,
    float *scratch,
    int in_channels, int out_channels) // scratch: 832B per OC_BLOCK
{
  unsigned long long t = __rdtsc();

  assert(PAD <= (K - 1)/2);
  assert(ncolblocks >= 1);

  const int WOUT = WIDTH + 2*PAD - K + 1;
  const int ALIGNED_W = (WOUT + 16 - 1)/16*16;

#ifdef __AVX512F__
  const int REG_BLOCK_SIZE = 30; // use at most 30 SIMD registers out of 32
#else
  const int REG_BLOCK_SIZE = 14; // use at most 14 SIMD registers out of 16
#endif

  const int REG_BLOCK_W = (WOUT + VLEN - 1)/VLEN;
  assert(REG_BLOCK_W <= REG_BLOCK_SIZE);
  const int REG_BLOCK_H = WOUT < REG_BLOCK_SIZE/REG_BLOCK_W ? WOUT : REG_BLOCK_SIZE/REG_BLOCK_W;
  // WIDTH = 13 (AlexNet conv3-5), AVX2 : REG_BLOCK_W = 2, REG_BLOCK_H = 7, ALIGNED_W = 16
  // WIDTH = 56 (GoogLeNet), AVX2 : REG_BLOCK_W = 7, REG_BLOCK_H = 2, ALIGNED_W = 64

#ifdef __AVX512F__
  __mmask16 mask_v = (1 << (WOUT%VLEN)) - 1;
#else
  __attribute__ ((aligned(64))) int mask_temp[VLEN] = { 0 };
  for (int i = 0; i < WOUT%VLEN; ++i) {
    mask_temp[i] = -1;
  }
  SIMDITYPE mask_v = _MM_LOAD_SI((SIMDITYPE *)mask_temp);
#endif

  if (ncolblocks > 1) {
    for (int oc_begin = output_channel_begin; oc_begin < output_channel_end; oc_begin += OC_BLOCK) {
      int oc_end = std::min(oc_begin + OC_BLOCK, output_channel_end);

      SIMDFPTYPE sum[REG_BLOCK_H][REG_BLOCK_W];
      SIMDFPTYPE w_v;
      int off;

      const int *rowptr = rowptr_blocked[0];
      const int *colidx = colidx_blocked[0];
      const float *values = values_blocked[0];

      for (int oc = oc_begin; oc < oc_end; ++oc) {
        SIMDFPTYPE bias_v = _MM_SET1(bias[oc]);

        int jbegin = rowptr[oc];
        int jend = rowptr[oc + 1];

        // register blocking over input image positions
        int hbegin;
        for (hbegin = 0; hbegin < WOUT/REG_BLOCK_H*REG_BLOCK_H; hbegin += REG_BLOCK_H) {
          int hend = hbegin + REG_BLOCK_H;

#pragma unroll(REG_BLOCK_H) // compiler gives warning for unroll pragma, but it still unrolls as we want.
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = bias_v;

//#define DBG_SCONV
#ifdef DBG_SCONV
#define CHANNEL_TO_DEBUG (248)
#define ROW_TO_DEBUG (12)
#define COL_TO_DEBUG (12)
              if (oc == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG && COL_TO_DEBUG >= w*VLEN && COL_TO_DEBUG < (w + 1)*VLEN) {
                float temp[VLEN];
                _MM_STORE(temp, bias_v);
                printf("%g", temp[COL_TO_DEBUG - w*VLEN]);
              }
#endif
            }
          }

#define SCONV_INNER_PROD \
          for (int j = jbegin; j < jend; ++j) { \
            w_v = _MM_SET1(values[j]); \
            off = colidx[j]; \
   \
_Pragma("unroll(REG_BLOCK_H)") \
            for (int h = 0; h < REG_BLOCK_H; ++h) { /* by some reason, iterating from hbegin to hend prevents icc from unrolling */ \
_Pragma("unroll(REG_BLOCK_W") \
              for (int w = 0; w < REG_BLOCK_W; ++w) { \
                sum[h][w] = _MM_FMADD(w_v, _MM_LOADU(input + off + (h + hbegin)*(WIDTH + PAD) + VLEN*w), sum[h][w]); \
              } \
   \
              /*if (oc == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG) { \
                float temp[VLEN]; \
                _MM_STORE(temp, sum[h - hbegin][COL_TO_DEBUG/VLEN]); \
                printf(" + %g*%d:%g:%g", values[j], off, input[off + ROW_TO_DEBUG*(WIDTH + PAD) + COL_TO_DEBUG], temp[COL_TO_DEBUG%VLEN]); \
              }*/ \
            } \
          }

          SCONV_INNER_PROD;

#pragma unroll(REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _MM_STORE(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
            }
          }
        } // for each register block

        // remainder register block
        if (WOUT%REG_BLOCK_H != 0) {
          // Lower half of images
          int hend = WOUT;

#pragma unroll(WOUT%REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = bias_v;
            }
          }

#define SCONV_INNER_PROD_REMAINDER \
          for (int j = jbegin; j < jend; ++j) { \
            w_v = _MM_SET1(values[j]); \
            off = colidx[j]; \
   \
_Pragma("unroll(WOUT%REG_BLOCK_H)") \
            for (int h = hbegin; h < hend; ++h) { \
_Pragma("unroll(REG_BLOCK_W)") \
              for (int w = 0; w < REG_BLOCK_W; ++w) { \
                sum[h - hbegin][w] = _MM_FMADD(w_v, _MM_LOADU(input + off + h*(WIDTH + PAD) + VLEN*w), sum[h - hbegin][w]); \
                assert(off + h*(WIDTH + PAD) + VLEN*w + VLEN <= in_channels*(WIDTH + PAD)*(WIDTH + PAD) + PAD*(WIDTH + 2*PAD) + VLEN - 1); \
              } \
            } \
          }

          SCONV_INNER_PROD_REMAINDER;

#pragma unroll(WOUT%REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              _MM_STORE(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
            }
          }
        } // remainder register block
      } // for each output channel

      for (int b = 1; b < ncolblocks - 1; ++b) {
        rowptr = rowptr_blocked[b];
        colidx = colidx_blocked[b];
        values = values_blocked[b];

        for (int oc = oc_begin; oc < oc_end; ++oc) {
          int jbegin = rowptr[oc];
          int jend = rowptr[oc + 1];

          // register blocking over input image positions
          int hbegin;
          for (hbegin = 0; hbegin < WOUT/REG_BLOCK_H*REG_BLOCK_H; hbegin += REG_BLOCK_H) {
            int hend = hbegin + REG_BLOCK_H;

#pragma unroll(REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
              for (int w = 0; w < REG_BLOCK_W; ++w) {
                sum[h - hbegin][w] = _MM_LOAD(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
              }
            }

            SCONV_INNER_PROD;

#pragma unroll(REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
              for (int w = 0; w < REG_BLOCK_W; ++w) {
                _MM_STORE(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
              }
            }
          } // for each register block

          // remainder register block
          if (WOUT%REG_BLOCK_H != 0) {
            int hend = WOUT;

#pragma unroll(WOUT%REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
              for (int w = 0; w < REG_BLOCK_W; ++w) {
                sum[h - hbegin][w] = _MM_LOAD(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
              }
            }

            SCONV_INNER_PROD_REMAINDER;

#pragma unroll(WOUT%REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
              for (int w = 0; w < REG_BLOCK_W; ++w) {
                _MM_STORE(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w, sum[h - hbegin][w]);
              }
            }
          } // remainder register block
        } // for each output channel
      } // for each col block

      rowptr = rowptr_blocked[ncolblocks - 1];
      colidx = colidx_blocked[ncolblocks - 1];
      values = values_blocked[ncolblocks - 1];

      for (int oc = oc_begin; oc < oc_end; ++oc) {
        int jbegin = rowptr[oc];
        int jend = rowptr[oc + 1];

        // register blocking over input image positions
        int hbegin;
        for (hbegin = 0; hbegin < WOUT/REG_BLOCK_H*REG_BLOCK_H; hbegin += REG_BLOCK_H) {
          int hend = hbegin + REG_BLOCK_H;

#pragma unroll(REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = _MM_LOAD(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
            }
          }

          SCONV_INNER_PROD;

          if (FUSE_RELU) {
#pragma unroll(REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
              if (WOUT%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
                for (int w = 0; w < REG_BLOCK_W; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
                }

#ifdef DBG_SCONV
                if (oc == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG) {
                  printf(" = %g\n", output[(oc*WOUT + h)*WOUT + COL_TO_DEBUG]);
                }
#endif
              }
              else {
                int w;
#pragma unroll(REG_BLOCK_W - 1)
                for (w = 0; w < REG_BLOCK_W - 1; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
                }
                _MM_MASK_STORE(output + (oc*WOUT + h)*WOUT + VLEN*w, mask_v, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
              }
            }
          } // FUSE_RELU
          else {
#pragma unroll(REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
              if (WOUT%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
                for (int w = 0; w < REG_BLOCK_W; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
                }

#ifdef DBG_SCONV
                if (oc == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG) {
                  printf(" = %g\n", output[(oc*WOUT + h)*WOUT + COL_TO_DEBUG]);
                }
#endif
              }
              else {
                int w;
  #pragma unroll(REG_BLOCK_W - 1)
                for (w = 0; w < REG_BLOCK_W - 1; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
                }
                _MM_MASK_STORE(output + (oc*WOUT + h)*WOUT + VLEN*w, mask_v, sum[h - hbegin][w]);
              }
            }
          } // !FUSE_RELU
        }

        // remainder register block
        if (WOUT%REG_BLOCK_H != 0) {
          int hend = WOUT;

#pragma unroll(WOUT%REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = _MM_LOAD(scratch + ((oc - oc_begin)*WOUT + h)*ALIGNED_W + VLEN*w);
            }
          }

          SCONV_INNER_PROD_REMAINDER;

          if (FUSE_RELU) {
#pragma unroll(WOUT%REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
              if (WOUT%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
                for (int w = 0; w < REG_BLOCK_W; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
                }
              }
              else {
                int w;
#pragma unroll(REG_BLOCK_W - 1)
                for (w = 0; w < REG_BLOCK_W - 1; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
                }
                assert((oc*WOUT + h)*WOUT + VLEN*w + WOUT%VLEN <= out_channels*WOUT*WOUT);
                _MM_MASK_STORE(output + (oc*WOUT + h)*WOUT + VLEN*w, mask_v, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO())); // invalid memory access reported from inspector (called from conv_relu_pool layer)
              }
            }
          }
          else {
#pragma unroll(WOUT%REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
              if (WOUT%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
                for (int w = 0; w < REG_BLOCK_W; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
                }
              }
              else {
                int w;
#pragma unroll(REG_BLOCK_W - 1)
                for (w = 0; w < REG_BLOCK_W - 1; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
                }
                assert((oc*WOUT + h)*WOUT + VLEN*w + WOUT%VLEN <= out_channels*WOUT*WOUT);
                _MM_MASK_STORE(output + (oc*WOUT + h)*WOUT + VLEN*w, mask_v, sum[h - hbegin][w]); // invalid memory access reported from inspector (called from conv_relu_pool layer)
              }
            }
          } // !FUSE_RELU
        } // remainder register block
      } // for each output channel
    }
  } // ncolblocks > 1
  else {
    for (int oc_begin = output_channel_begin; oc_begin < output_channel_end; oc_begin += OC_BLOCK) {
      int oc_end = std::min(oc_begin + OC_BLOCK, output_channel_end);

      SIMDFPTYPE sum[REG_BLOCK_H][REG_BLOCK_W];
      SIMDFPTYPE w_v;
      int off;

      const int *rowptr = rowptr_blocked[0];
      const int *colidx = colidx_blocked[0];
      const float *values = values_blocked[0];

      for (int oc = oc_begin; oc < oc_end; ++oc) {
        SIMDFPTYPE bias_v = _MM_SET1(bias[oc]);

        int jbegin = rowptr[oc];
        int jend = rowptr[oc + 1];

        // register blocking over input image positions
        int hbegin;
        for (hbegin = 0; hbegin < WOUT/REG_BLOCK_H*REG_BLOCK_H; hbegin += REG_BLOCK_H) {
          int hend = hbegin + REG_BLOCK_H;

#pragma unroll(REG_BLOCK_H) // compiler gives warning for unroll pragma, but it still unrolls as we want.
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = bias_v;

//#define DBG_SCONV
#ifdef DBG_SCONV
#define CHANNEL_TO_DEBUG (248)
#define ROW_TO_DEBUG (12)
#define COL_TO_DEBUG (12)
              if (oc == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG && COL_TO_DEBUG >= w*VLEN && COL_TO_DEBUG < (w + 1)*VLEN) {
                float temp[VLEN];
                _MM_STORE(temp, bias_v);
                printf("%g", temp[COL_TO_DEBUG - w*VLEN]);
              }
#endif
            }
          }

#define SCONV_INNER_PROD \
          for (int j = jbegin; j < jend; ++j) { \
            w_v = _MM_SET1(values[j]); \
            off = colidx[j]; \
   \
_Pragma("unroll(REG_BLOCK_H)") \
            for (int h = 0; h < REG_BLOCK_H; ++h) { /* by some reason, iterating from hbegin to hend prevents icc from unrolling */ \
_Pragma("unroll(REG_BLOCK_W") \
              for (int w = 0; w < REG_BLOCK_W; ++w) { \
                sum[h][w] = _MM_FMADD(w_v, _MM_LOADU(input + off + (h + hbegin)*(WIDTH + PAD) + VLEN*w), sum[h][w]); \
              } \
   \
              /*if (oc == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG) { \
                float temp[VLEN]; \
                _MM_STORE(temp, sum[h - hbegin][COL_TO_DEBUG/VLEN]); \
                printf(" + %g*%d:%g:%g", values[j], off, input[off + ROW_TO_DEBUG*(WIDTH + PAD) + COL_TO_DEBUG], temp[COL_TO_DEBUG%VLEN]); \
              }*/ \
            } \
          }

          SCONV_INNER_PROD;

          if (FUSE_RELU) {
#pragma unroll(REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
              if (WOUT%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
                for (int w = 0; w < REG_BLOCK_W; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
                }

#ifdef DBG_SCONV
                if (oc == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG) {
                  printf(" = %g\n", output[(oc*WOUT + h)*WOUT + COL_TO_DEBUG]);
                }
#endif
              }
              else {
                int w;
#pragma unroll(REG_BLOCK_W - 1)
                for (w = 0; w < REG_BLOCK_W - 1; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
                }
                _MM_MASK_STORE(output + (oc*WOUT + h)*WOUT + VLEN*w, mask_v, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
              }
            }
          }
          else {
#pragma unroll(REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
              if (WOUT%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
                for (int w = 0; w < REG_BLOCK_W; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
                }

#ifdef DBG_SCONV
                if (oc == CHANNEL_TO_DEBUG && h == ROW_TO_DEBUG) {
                  printf(" = %g\n", output[(oc*WOUT + h)*WOUT + COL_TO_DEBUG]);
                }
#endif
              }
              else {
                int w;
#pragma unroll(REG_BLOCK_W - 1)
                for (w = 0; w < REG_BLOCK_W - 1; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
                }
                _MM_MASK_STORE(output + (oc*WOUT + h)*WOUT + VLEN*w, mask_v, sum[h - hbegin][w]);
              }
            }
          } // !FUSE_RELU
        } // for each register block

        // remainder register block
        if (WOUT%REG_BLOCK_H != 0) {
          // Lower half of images
          int hend = WOUT;

#pragma unroll(WOUT%REG_BLOCK_H)
          for (int h = hbegin; h < hend; ++h) {
#pragma unroll(REG_BLOCK_W)
            for (int w = 0; w < REG_BLOCK_W; ++w) {
              sum[h - hbegin][w] = bias_v;
            }
          }

          SCONV_INNER_PROD_REMAINDER;

          if (FUSE_RELU) {
#pragma unroll(WOUT%REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
              if (WOUT%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
                for (int w = 0; w < REG_BLOCK_W; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
                }
              }
              else {
                int w;
#pragma unroll(REG_BLOCK_W - 1)
                for (w = 0; w < REG_BLOCK_W - 1; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
                }
                _MM_MASK_STORE(output + (oc*WOUT + h)*WOUT + VLEN*w, mask_v, _MM_MAX(sum[h - hbegin][w], _MM_SETZERO()));
              }
            }
          } // FUSE_RELU
          else {
#pragma unroll(WOUT%REG_BLOCK_H)
            for (int h = hbegin; h < hend; ++h) {
              if (WOUT%VLEN == 0) {
#pragma unroll(REG_BLOCK_W)
                for (int w = 0; w < REG_BLOCK_W; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
                }
              }
              else {
                int w;
#pragma unroll(REG_BLOCK_W - 1)
                for (w = 0; w < REG_BLOCK_W - 1; ++w) {
                  _MM_STOREU(output + (oc*WOUT + h)*WOUT + VLEN*w, sum[h - hbegin][w]);
                }
                _MM_MASK_STORE(output + (oc*WOUT + h)*WOUT + VLEN*w, mask_v, sum[h - hbegin][w]);
              }
            }
          } // !FUSE_RELU
        } // remainder register block
      } // for each output channel
    }
  } // ncolblocks == 1

  conv_cycles_of_this_batch[omp_get_thread_num()*16] += __rdtsc() - t;
}

/**
 * Default un-optimized sparse convolution implementation fused with bias term
 */
template<bool FUSE_RELU = false>
void caffe_cpu_sconv_default(
    // input features
    const float *input_padded, int in_channels,
    int height, int width,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    // weights
    const int *rowptr, const int *colidx, const float *values,
    int kernel_h, int kernel_w,
    const float *bias,
    // output features
    float *output,
    int out_channels)
{
  const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  conv_cycles_of_this_batch[omp_get_thread_num()*16] = __rdtsc();

  if (dilation_h != 1 || dilation_w != 1) {
    for (int output_row = 0; output_row < output_h; ++output_row) {
      for (int output_col = 0; output_col < output_w; ++output_col) {

        for (int oc = 0; oc < out_channels; ++oc) {
          float sum = bias[oc];

          for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
            int off = colidx[j];

            int kernel_col = off%(width + pad_w);
            int kernel_row = (off/(width + pad_w))%(height + pad_h);
            int in_channel = off/((width + pad_w)*(height + pad_h));

            int input_row = kernel_row * dilation_h + output_row * stride_h;
            int input_col = kernel_col * dilation_w + output_col * stride_w;

            sum += values[j]*input_padded[(in_channel * (height + pad_h) + input_row) * (width + pad_w) + input_col];
          }

          output[(oc * output_h + output_row) * output_w + output_col] = FUSE_RELU ? std::max(0.f, sum) : sum;
        }
      }
    }
  }
  else {
    for (int output_row = 0; output_row < output_h; ++output_row) {
      for (int output_col = 0; output_col < output_w; ++output_col) {

        const float *in = input_padded + output_row * stride_h * (width + pad_w) + output_col * stride_w;

        for (int oc = 0; oc < out_channels; ++oc) {
          float sum = bias[oc];
#ifdef DBG_SCONV
          if (oc == CHANNEL_TO_DEBUG && output_row == ROW_TO_DEBUG && output_col == COL_TO_DEBUG) {
            printf("%g", sum);
          }
#endif

          for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
            assert(in + colidx[j] >= input_padded && in + colidx[j] < input_padded + in_channels*(width + pad_w)*(height + pad_h) + pad_h*(width + 2*pad_w));
            sum += values[j]*in[colidx[j]];
#ifdef DBG_SCONV
            if (oc == CHANNEL_TO_DEBUG && output_row == ROW_TO_DEBUG && output_col == COL_TO_DEBUG) {
              printf(" + %g*%d:%g:%g", values[j], colidx[j], in[colidx[j]], sum);
            }
#endif
          }

          output[(oc*output_h + output_row)*output_w + output_col] = FUSE_RELU ? std::max(0.f, sum) : sum;
#ifdef DBG_SCONV
          if (oc == CHANNEL_TO_DEBUG && output_row == ROW_TO_DEBUG && output_col == COL_TO_DEBUG) {
            printf(" = %g\n", sum);
          }
#endif
        }
      }
    }
  }

  conv_cycles_of_this_batch[omp_get_thread_num()*16] = __rdtsc() - conv_cycles_of_this_batch[omp_get_thread_num()*16];
}

#endif /* _CAFFE_UTIL_CONV_HPP_ */
