/* Number-theoretic transform over Z_3329 (CRYSTAL-Kyber), ported from APS.

   This is the NTT that CRYSTAL-Kyber (ML-KEM) uses for its degree-256
   polynomial arithmetic: q = 3329, n = 256, a primitive 256th root of unity
   zeta = 17, and seven radix-2 stages whose twiddle factors are read from a
   table indexed by the stage size and the block number.

   The kernel is a faithful port of the NTT used as APS's post-quantum
   cryptography case study (Xiao et al., "APS: Open-Source Hardware-Software
   Co-Design Framework for Agile Processor Specialization", ICCAD'25 invited
   paper, Sec. VI-C). APS publishes that kernel as

       void pure_ntt (volatile uint16_t *a)
       {
         uint32_t t = 256;
         for (uint32_t m = 1; m < 128; m <<= 1) {
           t >>= 1;
           for (uint32_t i = 0; i < m; i++) {
             uint32_t j1 = (i << 1) * t;
             uint16_t s = PRE_COMPUT_TABLE_NTT[m + i];
             for (uint32_t j = j1; j < j1 + t; j++) {
               uint32_t ys = (a[j + t] * s) % 3329;
               a[j + t] = (a[j] + 3329 - ys) % 3329;
               a[j] = (a[j] + ys) % 3329;
             }
           }
         }
       }

   and this file keeps its loop nest, its index arithmetic, its 16-bit
   coefficients and its "reduction by %" form, because that form is exactly
   what the APS hardware case study accelerates (their Butterfly ISAX replaces
   the inner body with a Barrett reduction). Two deviations, both deliberate
   and documented below:

     1. the volatile qualifier is dropped, so the scalar baseline here is
        ordinary optimized C rather than a memory-clobbered loop;
     2. one invocation transforms NTT_POLYS = 8 independent polynomials, one
        ntt_256 () call each, instead of a single polynomial. The unit of work
        is unchanged: every call is one 256-point forward NTT, byte-for-byte
        the kernel above, so per-transform cycles are the total divided by 8.

   PRE_COMPUT_TABLE_NTT[k] = 17 ^ bitrev7 (k) mod 3329, for k = 1..127
   (entry 0 is unused). Entry k is the twiddle of the k-th butterfly group,
   which is also the ordering of the Kyber reference implementation's
   Montgomery-domain zetas[] table. The table is precomputed, as its name says,
   and lives in read-only data, i.e. in the same place a Kyber binary keeps it.

   Embench-style interface: initialise_benchmark / warm_caches / benchmark /
   verify_benchmark.

   SPDX-License-Identifier: MIT */

#include "support.h"
#include <stdint.h>
#include <string.h>

/* This scale factor will be changed to equalise the runtime of the
   benchmarks. */
#define LOCAL_SCALE_FACTOR 1

#define NTT_N 256
#define NTT_Q 3329
#define NTT_POLYS 8

#define EXPECTED_RESULT 0xbe265793u

/* Twiddle factors, 17 ^ bitrev7 (k) mod 3329.  Same ordering and same values
   as the plain (non-Montgomery) form of the Kyber reference zetas[] table. */
static const uint16_t PRE_COMPUT_TABLE_NTT[128] = {
      1,  1729,  2580,  3289,  2642,   630,  1897,   848,  /*   0-  7 */
   1062,  1919,   193,   797,  2786,  3260,   569,  1746,  /*   8- 15 */
    296,  2447,  1339,  1476,  3046,    56,  2240,  1333,  /*  16- 23 */
   1426,  2094,   535,  2882,  2393,  2879,  1974,   821,  /*  24- 31 */
    289,   331,  3253,  1756,  1197,  2304,  2277,  2055,  /*  32- 39 */
    650,  1977,  2513,   632,  2865,    33,  1320,  1915,  /*  40- 47 */
   2319,  1435,   807,   452,  1438,  2868,  1534,  2402,  /*  48- 55 */
   2647,  2617,  1481,   648,  2474,  3110,  1227,   910,  /*  56- 63 */
     17,  2761,   583,  2649,  1637,   723,  2288,  1100,  /*  64- 71 */
   1409,  2662,  3281,   233,   756,  2156,  3015,  3050,  /*  72- 79 */
   1703,  1651,  2789,  1789,  1847,   952,  1461,  2687,  /*  80- 87 */
    939,  2308,  2437,  2388,   733,  2337,   268,   641,  /*  88- 95 */
   1584,  2298,  2037,  3220,   375,  2549,  2090,  1645,  /*  96-103 */
   1063,   319,  2773,   757,  2099,   561,  2466,  2594,  /* 104-111 */
   2804,  1092,   403,  1026,  1143,  2150,  2775,   886,  /* 112-119 */
   1722,  1212,  1874,  1029,  2110,  2935,   885,  2154,  /* 120-127 */
};

/* The input polynomials, built once and never modified; every measured pass
   transforms a fresh copy, so a run's output does not depend on how many warm
   up passes ran before it. */
static uint16_t polys[NTT_POLYS][NTT_N] __attribute__ ((aligned (4)));
static uint16_t work[NTT_POLYS][NTT_N] __attribute__ ((aligned (4)));

/* One 256-point forward NTT, in place.  Body identical to APS's pure_ntt (). */
static void __attribute__ ((noinline))
ntt_256 (uint16_t * a)
{
  uint32_t t = 256;

  for (uint32_t m = 1; m < 128; m <<= 1)
    {
      t >>= 1;
      for (uint32_t i = 0; i < m; i++)
	{
	  uint32_t j1 = (i << 1) * t;
	  uint16_t s = PRE_COMPUT_TABLE_NTT[m + i];

	  for (uint32_t j = j1; j < j1 + t; j++)
	    {
	      uint32_t ys = ((uint32_t) a[j + t] * s) % 3329;
	      a[j + t] = (uint16_t) (((uint32_t) a[j] + 3329 - ys) % 3329);
	      a[j] = (uint16_t) (((uint32_t) a[j] + ys) % 3329);
	    }
	}
    }
}

static uint32_t
output_digest (const uint16_t * a)
{
  uint32_t h = 2166136261u;

  for (int k = 0; k < NTT_POLYS * NTT_N; k++)
    {
      h ^= (uint8_t) (a[k] & 0xFFu);
      h *= 16777619u;
      h ^= (uint8_t) (a[k] >> 8);
      h *= 16777619u;
    }
  return h;
}

void
initialise_benchmark (void)
{
  uint32_t seed = 0x6B657962u;	/* "kybe" */

  for (int p = 0; p < NTT_POLYS; p++)
    for (int k = 0; k < NTT_N; k++)
      {
	seed = seed * 1664525u + 1013904223u;
	polys[p][k] = (uint16_t) ((seed >> 8) % NTT_Q);
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
      memcpy (work, polys, sizeof polys);

      for (int p = 0; p < NTT_POLYS; p++)
	ntt_256 (work[p]);

      result = output_digest (&work[0][0]);
    }

  return (int) result;
}

int
verify_benchmark (int res)
{
  /* Every transformed coefficient is covered, not only the digest. */
  return output_digest (&work[0][0]) == EXPECTED_RESULT
    && (uint32_t) res == EXPECTED_RESULT;
}

/*
   Local Variables:
   mode: C
   c-file-style: "gnu"
   End:
 */
