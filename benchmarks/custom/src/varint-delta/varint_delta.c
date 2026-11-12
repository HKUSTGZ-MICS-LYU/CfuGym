/* Delta + zigzag + varint decoding of packed sensor telemetry.

   Telemetry frames from a sensor gateway carry each channel as a packed
   sequence of signed deltas, ZigZag-mapped and stored as base-128 varints
   (LEB128): protobuf "packed sint32", the MQTT remaining-length field and
   most time-series stores (Prometheus remote write, Gorilla-style blocks)
   use the same encoding.  Each varint byte holds 7 payload bits, least
   significant group first; the top bit is set on every byte except the last.

   One benchmark invocation decodes N_CHANNELS frames (temperature,
   acceleration, timestamps and a counter with occasional large jumps, so
   the stream mixes 1..5-byte varints) back into absolute samples.

   Embench-style interface: initialise_benchmark / warm_caches / benchmark /
   verify_benchmark.

   SPDX-License-Identifier: MIT */

#include "support.h"
#include <stdint.h>

/* This scale factor will be changed to equalise the runtime of the
   benchmarks. */
#define LOCAL_SCALE_FACTOR 8

#define N_CHANNELS 4
#define N_SAMPLES 1024
#define MAX_VARINT_BYTES 5

#define EXPECTED_RESULT 0x031aa45bu

static uint8_t streams[N_CHANNELS][N_SAMPLES * MAX_VARINT_BYTES];
static uint32_t stream_len[N_CHANNELS];
static int32_t samples[N_CHANNELS][N_SAMPLES] __attribute__ ((aligned (4)));
static uint32_t expected_digest;

/* Decode n delta/zigzag/varint values from src into absolute samples.
   Returns the number of bytes consumed. */
static uint32_t __attribute__ ((noinline))
decode_channel (const uint8_t *src, int32_t *dst, int n)
{
  const uint8_t *p = src;
  uint32_t prev = 0;

  for (int i = 0; i < n; i++)
    {
      uint32_t v = 0;
      unsigned int shift = 0;
      uint8_t b;
      do
	{
	  b = *p++;
	  v |= (uint32_t) (b & 0x7F) << shift;
	  shift += 7;
	}
      while (b & 0x80);

      uint32_t delta = (v >> 1) ^ (0u - (v & 1u));	/* ZigZag decode */
      prev += delta;
      dst[i] = (int32_t) prev;
    }
  return (uint32_t) (p - src);
}

static uint32_t
digest_word (uint32_t h, uint32_t w)
{
  for (int k = 0; k < 4; k++)
    {
      h ^= (w >> (8 * k)) & 0xFFu;
      h *= 16777619u;
    }
  return h;
}

static uint32_t
samples_digest (void)
{
  uint32_t h = 2166136261u;
  for (int c = 0; c < N_CHANNELS; c++)
    for (int i = 0; i < N_SAMPLES; i++)
      h = digest_word (h, (uint32_t) samples[c][i]);
  return h;
}

static uint32_t
put_varint (uint8_t *dst, uint32_t v)
{
  uint32_t n = 0;
  while (v >= 0x80)
    {
      dst[n++] = (uint8_t) (v | 0x80);
      v >>= 7;
    }
  dst[n++] = (uint8_t) v;
  return n;
}

static uint32_t lcg_state;

static uint32_t
lcg (void)
{
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return lcg_state >> 8;
}

/* Next raw sample of channel c (host-side signal model). */
static uint32_t
next_sample (int c, int i, uint32_t prev)
{
  uint32_t r = lcg ();
  switch (c)
    {
    case 0:			/* temperature, centi-degrees: tiny random walk */
      return prev + (r % 7u) - 3u;
    case 1:			/* accelerometer: triangle wave + noise */
      {
	int32_t phase = (int32_t) (i & 127);
	int32_t tri = phase < 64 ? phase * 96 : (127 - phase) * 96;
	return (uint32_t) (tri - 3000 + (int32_t) (r % 129u) - 64);
      }
    case 2:			/* timestamps in microseconds, 10 ms period + jitter */
      return prev + 10000u + (r % 129u) - 64u;
    default:			/* counter with occasional resets / large jumps */
      if ((r & 63u) == 0)
	return (r << 7) ^ 0x9E3779B9u;
      return prev + (r % 50u);
    }
}

void
initialise_benchmark (void)
{
  uint32_t h = 2166136261u;
  lcg_state = 0x1234ABCDu;

  for (int c = 0; c < N_CHANNELS; c++)
    {
      uint32_t prev = 0, len = 0, cur = 0;
      for (int i = 0; i < N_SAMPLES; i++)
	{
	  cur = next_sample (c, i, cur);
	  uint32_t d = cur - prev;
	  uint32_t zz = (d << 1) ^ (0u - (d >> 31));	/* ZigZag encode */
	  len += put_varint (&streams[c][len], zz);
	  prev = cur;
	  samples[c][i] = 0;
	}
      stream_len[c] = len;
    }

  /* Digest of the decoded samples a correct decoder must reproduce. */
  lcg_state = 0x1234ABCDu;
  for (int c = 0; c < N_CHANNELS; c++)
    {
      uint32_t cur = 0;
      for (int i = 0; i < N_SAMPLES; i++)
	{
	  cur = next_sample (c, i, cur);
	  h = digest_word (h, cur);
	}
    }
  expected_digest = h;
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
      uint32_t framing_ok = 1;
      for (int c = 0; c < N_CHANNELS; c++)
	{
	  uint32_t used = decode_channel (streams[c], samples[c], N_SAMPLES);
	  framing_ok &= (used == stream_len[c]);
	}
      result = framing_ok ? samples_digest () : 0;
    }

  return (int) result;
}

int
verify_benchmark (int res)
{
  /* Every decoded sample is checked against the encoder's signal. */
  uint32_t h = samples_digest ();
  return h == expected_digest && h == EXPECTED_RESULT
    && (uint32_t) res == EXPECTED_RESULT;
}

/*
   Local Variables:
   mode: C
   c-file-style: "gnu"
   End:
*/
