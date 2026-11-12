/* 3x3 median filter for an 8-bit grayscale sensor frame.

   Impulse ("salt-and-pepper") noise from a low-cost CMOS camera or a thermal
   array is removed with a 3x3 median filter before the frame is analysed.
   This is the straightforward reference implementation: gather the 3x3
   neighbourhood of every interior pixel, sort it with an insertion sort and
   keep the middle element.  Border pixels are copied unchanged.

   Embench-style interface: initialise_benchmark / warm_caches / benchmark /
   verify_benchmark.

   SPDX-License-Identifier: MIT */

#include "support.h"
#include <stdint.h>

/* This scale factor will be changed to equalise the runtime of the
   benchmarks. */
#define LOCAL_SCALE_FACTOR 2

#define IMG_W 64
#define IMG_H 32

#define EXPECTED_RESULT 0x66cf74d9u

static uint8_t img_in[IMG_H][IMG_W] __attribute__ ((aligned (4)));
static uint8_t img_out[IMG_H][IMG_W] __attribute__ ((aligned (4)));

/* Median of nine samples: sort the window, return the middle element. */
static uint8_t
median9 (uint8_t *w)
{
  for (int i = 1; i < 9; i++)
    {
      uint8_t v = w[i];
      int j = i - 1;
      while (j >= 0 && w[j] > v)
	{
	  w[j + 1] = w[j];
	  j--;
	}
      w[j + 1] = v;
    }
  return w[4];
}

static void __attribute__ ((noinline))
median_filter_3x3 (const uint8_t *in, uint8_t *out, int width, int height)
{
  /* Borders are passed through. */
  for (int x = 0; x < width; x++)
    {
      out[x] = in[x];
      out[(height - 1) * width + x] = in[(height - 1) * width + x];
    }
  for (int y = 1; y < height - 1; y++)
    {
      out[y * width] = in[y * width];
      out[y * width + width - 1] = in[y * width + width - 1];
    }

  for (int y = 1; y < height - 1; y++)
    for (int x = 1; x < width - 1; x++)
      {
	uint8_t window[9];
	int k = 0;
	for (int dy = -1; dy <= 1; dy++)
	  for (int dx = -1; dx <= 1; dx++)
	    window[k++] = in[(y + dy) * width + (x + dx)];
	out[y * width + x] = median9 (window);
      }
}

static uint32_t
frame_digest (void)
{
  uint32_t h = 2166136261u;
  for (int y = 0; y < IMG_H; y++)
    for (int x = 0; x < IMG_W; x++)
      {
	h ^= img_out[y][x];
	h *= 16777619u;
      }
  return h;
}

void
initialise_benchmark (void)
{
  /* Smooth scene (gradient + soft texture) with ~10% impulse noise. */
  uint32_t seed = 0x6C8E9CF5u;
  for (int y = 0; y < IMG_H; y++)
    for (int x = 0; x < IMG_W; x++)
      {
	uint32_t v = 40u + (uint32_t) (x * 2 + y * 3)
	  + (uint32_t) (((x ^ y) & 7) * 4) + (uint32_t) (((x * y) >> 4) & 15);
	seed = seed * 1664525u + 1013904223u;
	uint32_t r = seed >> 24;
	if (r < 13)
	  v = 0;
	else if (r < 26)
	  v = 255;
	else if (v > 255)
	  v = 255;
	img_in[y][x] = (uint8_t) v;
	img_out[y][x] = 0;
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
      median_filter_3x3 (&img_in[0][0], &img_out[0][0], IMG_W, IMG_H);
      result = frame_digest ();
    }

  return (int) result;
}

int
verify_benchmark (int res)
{
  /* The whole filtered frame is checked, not only the returned digest. */
  return frame_digest () == EXPECTED_RESULT
    && (uint32_t) res == EXPECTED_RESULT;
}

/*
   Local Variables:
   mode: C
   c-file-style: "gnu"
   End:
*/
