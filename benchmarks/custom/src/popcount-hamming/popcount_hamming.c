/* Near-duplicate search over binary fingerprints: Hamming distance by
   population count.

   Each record is reduced to a 512-bit binary signature (the simhash / LSH
   scheme used for web-page and image deduplication, and for binary embeddings
   in vector search).  A query signature is compared against every stored
   signature and the nearest one wins; signatures within a radius also count
   as hits.  Distances are computed word by word with a byte-wide popcount
   lookup table, which is how portable embedded C counts bits without a hardware
   instruction.

   The same inner loop is the hot path of bitmap-index cardinality (Roaring /
   EWAH bitmaps), BCH and LDPC decoding, packet-mask classification, chess
   bitboards and Hamming-distance sketches - anywhere a 32-bit word is really a
   set of 32 flags.

   Embench-style interface: initialise_benchmark / warm_caches / benchmark /
   verify_benchmark.

   SPDX-License-Identifier: MIT */

#include "support.h"
#include <stdint.h>

/* This scale factor will be changed to equalise the runtime of the
   benchmarks. */
#define LOCAL_SCALE_FACTOR 4

#define N_SIG 256
#define W_WORDS 16			/* 512-bit signature */
#define N_CLUSTERS 16
#define CLUSTER_SIZE (N_SIG / N_CLUSTERS)
#define HIT_THRESHOLD 200		/* radius that keeps a near-duplicate cluster */

#define EXPECTED_RESULT 0x71e21b9bu
#define EXPECTED_DIGEST 0x71f21bc8u

static uint32_t signatures[N_SIG][W_WORDS] __attribute__ ((aligned (4)));
static uint32_t query[W_WORDS] __attribute__ ((aligned (4)));
static uint8_t pop8[256];

/* Nearest-signature scan.  Returns the hit count and reports the minimum
   distance. */
static uint32_t __attribute__ ((noinline))
hamming_scan (const uint32_t *sigs, const uint32_t *q, int n, uint32_t *best)
{
  uint32_t min_dist = 0xFFFFFFFFu;
  uint32_t best_index = 0;
  uint32_t hits = 0;

  for (int s = 0; s < n; s++)
    {
      const uint32_t *sig = sigs + s * W_WORDS;
      uint32_t dist = 0;

      for (int w = 0; w < W_WORDS; w++)
	{
	  uint32_t d = sig[w] ^ q[w];
	  dist += pop8[d & 0xFF]
	    + pop8[(d >> 8) & 0xFF]
	    + pop8[(d >> 16) & 0xFF] + pop8[(d >> 24) & 0xFF];
	}

      if (dist < min_dist)
	{
	  min_dist = dist;
	  best_index = (uint32_t) s;
	}
      if (dist <= HIT_THRESHOLD)
	hits++;
    }

  *best = best_index;
  return (hits << 16) ^ min_dist;
}

static uint32_t
signature_digest (void)
{
  uint32_t h = 2166136261u;
  for (int s = 0; s < N_SIG; s++)
    for (int w = 0; w < W_WORDS; w++)
      {
	uint32_t v = signatures[s][w];
	for (int k = 0; k < 4; k++)
	  {
	    h ^= (v >> (8 * k)) & 0xFFu;
	    h *= 16777619u;
	  }
      }
  return h;
}

void
initialise_benchmark (void)
{
  for (int i = 0; i < 256; i++)
    {
      uint8_t v = (uint8_t) i;
      uint8_t c = 0;
      while (v)
	{
	  c += (uint8_t) (v & 1u);
	  v >>= 1;
	}
      pop8[i] = c;
    }

  /* A corpus of near-duplicate clusters, as a deduplicated web/image index
     looks: each cluster has a centroid and members a few bits away from it,
     while clusters are far apart (>200 bits of 512). */
  uint32_t seed = 0x9E3779B9u;
  for (int c = 0; c < N_CLUSTERS; c++)
    {
      uint32_t centroid[W_WORDS];
      for (int w = 0; w < W_WORDS; w++)
	{
	  seed = seed * 1664525u + 1013904223u;
	  uint32_t v = seed;
	  uint32_t runs = (seed >> 13) & 7u;
	  for (uint32_t r = 0; r < runs; r++)
	    {
	      seed = seed * 1664525u + 1013904223u;
	      v |= 1u << ((seed >> 8) & 31u);
	    }
	  centroid[w] = v;
	}
      for (int m = 0; m < CLUSTER_SIZE; m++)
	{
	  int s = c * CLUSTER_SIZE + m;
	  for (int w = 0; w < W_WORDS; w++)
	    signatures[s][w] = centroid[w];
	  uint32_t flips = 1u + ((uint32_t) m & 3u);
	  for (uint32_t f = 0; f < flips; f++)
	    {
	      seed = seed * 1664525u + 1013904223u;
	      signatures[s][(seed >> 8) % W_WORDS] ^= 1u << ((seed >> 3) & 31u);
	    }
	}
    }

  /* Query: a perturbed copy of one cluster member. */
  for (int w = 0; w < W_WORDS; w++)
    query[w] = signatures[5 * CLUSTER_SIZE][w];
  for (int f = 0; f < 3; f++)
    {
      seed = seed * 1664525u + 1013904223u;
      query[(seed >> 8) % W_WORDS] ^= 1u << ((seed >> 3) & 31u);
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
      uint32_t best = 0;
      uint32_t packed = hamming_scan (&signatures[0][0], query, N_SIG, &best);
      /* The result depends on the scan and on the corpus it scanned. */
      result = packed ^ best ^ signature_digest ();
    }

  return (int) result;
}

int
verify_benchmark (int res)
{
  /* The returned value already binds the scan result, the winning index and
     the corpus digest; the digest is also checked directly, so a mutated
     corpus cannot be reported as a pass. */
  return signature_digest () == EXPECTED_DIGEST
    && (uint32_t) res == EXPECTED_RESULT;
}

/*
   Local Variables:
   mode: C
   c-file-style: "gnu"
   End:
*/
