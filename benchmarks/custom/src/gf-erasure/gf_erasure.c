/* Reed-Solomon style erasure coding over GF(2^8).

   A storage/telemetry node splits a block into K_DATA data shards and adds
   M_PARITY parity shards so that any M_PARITY lost shards can be rebuilt
   (the scheme used by RAID-6, Ceph/HDFS erasure-coded pools and many
   sensor-network FEC links).  The generator is systematic: identity rows for
   the data shards followed by a Cauchy matrix over GF(2^8) with the field
   polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D).

   Field multiplication uses the classic log/antilog tables, as in most
   portable C erasure-coding libraries.

   One benchmark invocation encodes the parity shards, then simulates the loss
   of two data shards and reconstructs them from the survivors (inverting the
   surviving generator rows with Gauss-Jordan elimination).

   Embench-style interface: initialise_benchmark / warm_caches / benchmark /
   verify_benchmark.

   SPDX-License-Identifier: MIT */

#include "support.h"
#include <stdint.h>
#include <string.h>

/* This scale factor will be changed to equalise the runtime of the
   benchmarks. */
#define LOCAL_SCALE_FACTOR 4

#define K_DATA 8
#define M_PARITY 4
#define N_SHARDS (K_DATA + M_PARITY)
#define SHARD_BYTES 256
#define GF_POLY 0x11D

#define LOST_A 2
#define LOST_B 5

#define EXPECTED_RESULT 0x95b629a2u

static uint8_t gf_log[256];
static uint8_t gf_exp[512];

static uint8_t data_shards[K_DATA][SHARD_BYTES] __attribute__ ((aligned (4)));
static uint8_t parity_shards[M_PARITY][SHARD_BYTES] __attribute__ ((aligned (4)));
static uint8_t recovered[2][SHARD_BYTES] __attribute__ ((aligned (4)));
static uint8_t enc_matrix[M_PARITY][K_DATA];
static uint8_t dec_matrix[2][K_DATA];

static inline uint8_t
gf_mul (uint8_t a, uint8_t b)
{
  if (a == 0 || b == 0)
    return 0;
  return gf_exp[gf_log[a] + gf_log[b]];
}

static inline uint8_t
gf_inv (uint8_t a)
{
  /* a != 0 */
  return gf_exp[255 - gf_log[a]];
}

/* dst[r][i] = XOR_j matrix[r][j] * src[j][i]  for r < rows, j < k.
   The core of both encoding and reconstruction. */
static void __attribute__ ((noinline))
ec_encode_region (int rows, int k, const uint8_t *matrix,
		  const uint8_t *const *src, uint8_t *const *dst, int len)
{
  for (int r = 0; r < rows; r++)
    {
      const uint8_t *coef = matrix + r * k;
      uint8_t *out = dst[r];
      for (int i = 0; i < len; i++)
	{
	  uint8_t acc = 0;
	  for (int j = 0; j < k; j++)
	    acc ^= gf_mul (coef[j], src[j][i]);
	  out[i] = acc;
	}
    }
}

/* Row 'row' of the systematic generator [I; C]. */
static void
generator_row (int row, uint8_t *out)
{
  if (row < K_DATA)
    {
      for (int j = 0; j < K_DATA; j++)
	out[j] = (uint8_t) (j == row);
    }
  else
    memcpy (out, enc_matrix[row - K_DATA], K_DATA);
}

/* Invert a K_DATA x K_DATA matrix in place (Gauss-Jordan).  Returns 0 on a
   singular matrix, which a Cauchy-based generator never produces. */
static int
gf_invert_matrix (uint8_t m[K_DATA][K_DATA], uint8_t inv[K_DATA][K_DATA])
{
  for (int r = 0; r < K_DATA; r++)
    for (int c = 0; c < K_DATA; c++)
      inv[r][c] = (uint8_t) (r == c);

  for (int col = 0; col < K_DATA; col++)
    {
      int pivot = col;
      while (pivot < K_DATA && m[pivot][col] == 0)
	pivot++;
      if (pivot == K_DATA)
	return 0;
      if (pivot != col)
	for (int c = 0; c < K_DATA; c++)
	  {
	    uint8_t t = m[col][c];
	    m[col][c] = m[pivot][c];
	    m[pivot][c] = t;
	    t = inv[col][c];
	    inv[col][c] = inv[pivot][c];
	    inv[pivot][c] = t;
	  }
      uint8_t scale = gf_inv (m[col][col]);
      for (int c = 0; c < K_DATA; c++)
	{
	  m[col][c] = gf_mul (m[col][c], scale);
	  inv[col][c] = gf_mul (inv[col][c], scale);
	}
      for (int r = 0; r < K_DATA; r++)
	{
	  if (r == col || m[r][col] == 0)
	    continue;
	  uint8_t f = m[r][col];
	  for (int c = 0; c < K_DATA; c++)
	    {
	      m[r][c] ^= gf_mul (f, m[col][c]);
	      inv[r][c] ^= gf_mul (f, inv[col][c]);
	    }
	}
    }
  return 1;
}

static uint32_t
fnv1a (uint32_t h, const uint8_t *p, int len)
{
  for (int i = 0; i < len; i++)
    {
      h ^= p[i];
      h *= 16777619u;
    }
  return h;
}

void
initialise_benchmark (void)
{
  /* Antilog/log tables for GF(2^8) with generator 2. */
  uint32_t x = 1;
  for (int i = 0; i < 255; i++)
    {
      gf_exp[i] = (uint8_t) x;
      gf_log[x] = (uint8_t) i;
      x <<= 1;
      if (x & 0x100)
	x ^= GF_POLY;
    }
  for (int i = 255; i < 512; i++)
    gf_exp[i] = gf_exp[i - 255];
  gf_log[0] = 0;

  /* Cauchy parity rows: C[i][j] = 1 / (x_i + y_j), x_i = K_DATA + i, y_j = j. */
  for (int i = 0; i < M_PARITY; i++)
    for (int j = 0; j < K_DATA; j++)
      enc_matrix[i][j] = gf_inv ((uint8_t) ((K_DATA + i) ^ j));

  /* Data shards: a deterministic, mildly structured byte stream. */
  uint32_t seed = 0x2545F491u;
  for (int d = 0; d < K_DATA; d++)
    for (int i = 0; i < SHARD_BYTES; i++)
      {
	seed = seed * 1103515245u + 12345u;
	data_shards[d][i] = (uint8_t) ((seed >> 16) ^ (i * (d + 1)));
      }
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
      /* 1. Encode: parity = C * data. */
      const uint8_t *src[N_SHARDS];
      uint8_t *dst[M_PARITY];
      for (int j = 0; j < K_DATA; j++)
	src[j] = data_shards[j];
      for (int p = 0; p < M_PARITY; p++)
	dst[p] = parity_shards[p];
      ec_encode_region (M_PARITY, K_DATA, &enc_matrix[0][0], src, dst,
			SHARD_BYTES);

      /* 2. Lose two data shards; pick the first K_DATA survivors. */
      int survivors[K_DATA];
      int n = 0;
      for (int s = 0; s < N_SHARDS && n < K_DATA; s++)
	if (s != LOST_A && s != LOST_B)
	  survivors[n++] = s;

      uint8_t m[K_DATA][K_DATA];
      uint8_t inv[K_DATA][K_DATA];
      for (int r = 0; r < K_DATA; r++)
	generator_row (survivors[r], m[r]);
      if (!gf_invert_matrix (m, inv))
	return 0;
      memcpy (dec_matrix[0], inv[LOST_A], K_DATA);
      memcpy (dec_matrix[1], inv[LOST_B], K_DATA);

      /* 3. Reconstruct the lost shards from the survivors. */
      for (int j = 0; j < K_DATA; j++)
	{
	  int s = survivors[j];
	  src[j] = s < K_DATA ? data_shards[s] : parity_shards[s - K_DATA];
	}
      uint8_t *rec[2] = { recovered[0], recovered[1] };
      ec_encode_region (2, K_DATA, &dec_matrix[0][0], src, rec, SHARD_BYTES);

      uint32_t h = 2166136261u;
      h = fnv1a (h, &parity_shards[0][0], M_PARITY * SHARD_BYTES);
      h = fnv1a (h, &recovered[0][0], 2 * SHARD_BYTES);
      result = h;
    }

  return (int) result;
}

int
verify_benchmark (int res)
{
  /* The rebuilt shards must equal the originals byte for byte. */
  if (memcmp (recovered[0], data_shards[LOST_A], SHARD_BYTES) != 0)
    return 0;
  if (memcmp (recovered[1], data_shards[LOST_B], SHARD_BYTES) != 0)
    return 0;
  /* Parity bytes and the returned digest are checked against a golden value. */
  uint32_t h = 2166136261u;
  h = fnv1a (h, &parity_shards[0][0], M_PARITY * SHARD_BYTES);
  h = fnv1a (h, &recovered[0][0], 2 * SHARD_BYTES);
  return h == EXPECTED_RESULT && (uint32_t) res == EXPECTED_RESULT;
}

/*
   Local Variables:
   mode: C
   c-file-style: "gnu"
   End:
*/
