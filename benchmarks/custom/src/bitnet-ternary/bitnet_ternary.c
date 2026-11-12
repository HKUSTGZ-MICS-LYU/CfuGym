/* BitNet-style ternary-weight matrix multiplication (BitNet b1.58), I2_S layout.

   In a BitNet b1.58 model the weights are ternary, {-1, 0, +1}, i.e. log2(3)
   = 1.58 bits each.  They are packed once, offline, when the model is
   quantized and exported; the device never packs anything.  This benchmark
   uses the "I2_S" layout of bitnet.cpp: two bits per weight, four weights per
   byte, with the codes {0, 1, 2} standing for {-1, 0, +1} (code 3 is unused).
   That is the portable, lossless baseline kernel used on every platform
   bitnet.cpp supports; the tuned TL1/TL2 kernels go further and use the packed
   code as an index into a per-block table of activation partial sums, so they
   never unpack a weight at all.

   Activations are int8, accumulators int32, then a per-row integer re-scale
   and saturation back to int8.

   The packer below stands in for the export-time quantizer: it runs in
   initialise_benchmark(), outside the timed region, and produces exactly the
   bytes a model file would contain.  Only reading and decoding them is timed.

   Embench-style interface: initialise_benchmark / warm_caches / benchmark /
   verify_benchmark.

   SPDX-License-Identifier: MIT */

#include "support.h"
#include <stdint.h>

/* This scale factor will be changed to equalise the runtime of the
   benchmarks. */
#define LOCAL_SCALE_FACTOR 2

#define M_ROWS 32
#define K_COLS 120
#define N_VECS 16
#define WEIGHTS_PER_BYTE 4	/* I2_S: 2 bits each */
#define PACKED_BYTES (K_COLS / WEIGHTS_PER_BYTE)	/* 30 */

#define EXPECTED_RESULT 0x4e2d8935u

static uint8_t packed_w[M_ROWS][PACKED_BYTES];
static int8_t acts[N_VECS][K_COLS] __attribute__ ((aligned (4)));
static int8_t out[M_ROWS][N_VECS] __attribute__ ((aligned (4)));
static uint8_t row_shift[M_ROWS];

static int8_t
sat8 (int32_t v)
{
  if (v > 127)
    return 127;
  if (v < -128)
    return -128;
  return (int8_t) v;
}

/* y[r][v] = sat8( (sum_k w[r][k] * x[v][k]) >> shift[r] ), with w unpacked from
   the packed I2_S row and x int8. */
static void __attribute__ ((noinline))
ternary_matmul (const uint8_t *w, const int8_t *x, int8_t *y,
		int rows, int kcols, int vecs, const uint8_t *shift)
{
  int bytes = kcols / WEIGHTS_PER_BYTE;

  for (int v = 0; v < vecs; v++)
    {
      const int8_t *xv = x + v * kcols;
      for (int r = 0; r < rows; r++)
	{
	  const uint8_t *wr = w + r * bytes;
	  int32_t acc = 0;

	  for (int b = 0; b < bytes; b++)
	    {
	      uint32_t code = wr[b];
	      int k = b * WEIGHTS_PER_BYTE;
	      for (int t = 0; t < WEIGHTS_PER_BYTE; t++)
		{
		  int32_t trit = (int32_t) (code & 3u) - 1;	/* 0,1,2 -> -1,0,+1 */
		  code >>= 2;
		  acc += trit * (int32_t) xv[k + t];
		}
	    }
	  y[r * vecs + v] = sat8 (acc >> shift[r]);
	}
    }
}

static uint32_t
output_digest (void)
{
  uint32_t h = 2166136261u;
  for (int r = 0; r < M_ROWS; r++)
    for (int v = 0; v < N_VECS; v++)
      {
	h ^= (uint8_t) out[r][v];
	h *= 16777619u;
      }
  return h;
}

void
initialise_benchmark (void)
{
  uint32_t seed = 0x1F123BB5u;

  /* Export-time packing: four trits per byte, two bits each, low bits first. */
  for (int r = 0; r < M_ROWS; r++)
    for (int b = 0; b < PACKED_BYTES; b++)
      {
	uint32_t code = 0;
	for (int t = 0; t < WEIGHTS_PER_BYTE; t++)
	  {
	    seed = seed * 1664525u + 1013904223u;
	    uint32_t trit = (seed >> 16) % 3u;	/* 0,1,2 -> -1,0,+1 */
	    /* Slight sparsity: zero is more likely than +-1, as in b1.58. */
	    if (((seed >> 20) & 3u) == 0)
	      trit = 1;
	    code |= trit << (2 * t);
	  }
	packed_w[r][b] = (uint8_t) code;
      }

  for (int v = 0; v < N_VECS; v++)
    for (int k = 0; k < K_COLS; k++)
      {
	seed = seed * 1664525u + 1013904223u;
	acts[v][k] = (int8_t) (((seed >> 12) & 0xFFu) - 128);
      }

  for (int r = 0; r < M_ROWS; r++)
    row_shift[r] = (uint8_t) (6 + (r % 3));
}

static int benchmark_body (unsigned int lsf, unsigned int gsf);

void
warm_caches (int heat)
{
  int res = benchmark_body (1, heat);

  return;
}

int
benchmark (void)
{
  return benchmark_body (LOCAL_SCALE_FACTOR, GLOBAL_SCALE_FACTOR);
}

static int __attribute__ ((noinline))
benchmark_body (unsigned int lsf, unsigned int gsf)
{
  uint32_t result = 0;

  for (unsigned int rep = 0; rep < lsf * gsf; rep++)
    {
      ternary_matmul (&packed_w[0][0], &acts[0][0], &out[0][0],
		      M_ROWS, K_COLS, N_VECS, row_shift);
      result = output_digest ();
    }

  return (int) result;
}

int
verify_benchmark (int res)
{
  /* The whole quantized output tile is checked, not only the digest. */
  return output_digest () == EXPECTED_RESULT
    && (uint32_t) res == EXPECTED_RESULT;
}

/*
   Local Variables:
   mode: C
   c-file-style: "gnu"
   End:
*/
