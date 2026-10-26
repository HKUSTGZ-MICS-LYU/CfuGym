@LICENSE@
/* Host-owned split-stage Huffbench driver/oracle (huffbench_stages_v1).
 * Reference encoding is prepared before start_trigger(), never by DUT code.
 * Encode and decode are both timed, but decode receives an independent stream.
 * This hardens the original fixed fixture; it is not hidden-input evaluation
 * or an isolation boundary against arbitrary memory-corrupting native code.
 */
#include "support.h"
#include "huffbench.h"
#include <string.h>
#define LOCAL_SCALE_FACTOR 1

typedef uint32_t bits32;
typedef unsigned char byte;
@INPUT@

@REFERENCE_CODEC@

static struct huffbench_encoded huffbench_reference;
static struct huffbench_encoded huffbench_observed;
static struct huffbench_encoded huffbench_decode_input;
static byte huffbench_input[HUFFBENCH_TEST_SIZE];
static byte huffbench_decoded[HUFFBENCH_TEST_SIZE];
static int huffbench_reference_ready;
static int huffbench_completed;

void initialise_benchmark(void)
{
  huffbench_completed = 0;
  huffbench_reference_ready = huffbench_reference_encode(
      orig_data, HUFFBENCH_TEST_SIZE, &huffbench_reference);
}

static int __attribute__((noinline))
benchmark_body(unsigned int lsf, unsigned int gsf)
{
  huffbench_completed = 0;
  for (unsigned int local = 0; local < lsf; ++local)
    for (unsigned int global = 0; global < gsf; ++global)
      {
        /* Reset every observable field, not struct padding or stale output. */
        huffbench_observed.length = 0;
        huffbench_decode_input.length = huffbench_reference.length;
        for (size_t i = 0; i < 256; ++i)
          {
            huffbench_observed.code[i] = 0;
            huffbench_observed.clen[i] = 0;
            huffbench_decode_input.code[i] = huffbench_reference.code[i];
            huffbench_decode_input.clen[i] = huffbench_reference.clen[i];
          }
        for (size_t i = 0; i < HUFFBENCH_CAPACITY; ++i)
          {
            huffbench_observed.bytes[i] = 0;
            huffbench_decode_input.bytes[i] = huffbench_reference.bytes[i];
          }
        for (size_t i = 0; i < HUFFBENCH_TEST_SIZE; ++i)
          {
            huffbench_input[i] = orig_data[i];
          }
        if (!huffbench_reference_ready ||
            !huffbench_encode(huffbench_input, HUFFBENCH_TEST_SIZE,
                              &huffbench_observed))
          return -1;
        for (size_t i = 0; i < HUFFBENCH_TEST_SIZE; ++i)
          huffbench_decoded[i] = orig_data[i] ^ 0xffu;
        if (!huffbench_decode(&huffbench_decode_input, huffbench_decoded,
                              HUFFBENCH_TEST_SIZE))
          return -1;
        ++huffbench_completed;
      }
  return 0;
}

void warm_caches(int heat) { (void)benchmark_body(1, (unsigned int)heat); }
int benchmark(void)
{
  return benchmark_body(LOCAL_SCALE_FACTOR, GLOBAL_SCALE_FACTOR);
}

int verify_benchmark(int result)
{
  size_t frequencies[256] = {0};
  size_t actual_bits = 0, optimal_bits = 0, position = 0;
  if (result != 0 || !huffbench_reference_ready ||
      huffbench_completed != LOCAL_SCALE_FACTOR * GLOBAL_SCALE_FACTOR ||
      huffbench_observed.length == 0 ||
      huffbench_observed.length > HUFFBENCH_CAPACITY)
    return 0;
  for (size_t i = 0; i < HUFFBENCH_TEST_SIZE; ++i)
    {
      if (huffbench_decoded[i] != orig_data[i])
        return 0;
      ++frequencies[orig_data[i]];
    }
  for (size_t symbol = 0; symbol < 256; ++symbol)
    {
      unsigned int length = huffbench_observed.clen[symbol];
      uint32_t code = huffbench_observed.code[symbol];
      if ((!frequencies[symbol] && length != 0) ||
          (frequencies[symbol] && (length == 0 || length > 32)))
        return 0;
      if (length && length < 32 && (code >> length) != 0)
        return 0;
      actual_bits += frequencies[symbol] * length;
      optimal_bits += frequencies[symbol] * huffbench_reference.clen[symbol];
      /* Reject duplicate codes and all proper-prefix collisions. */
      if (length)
        for (size_t other = 0; other < symbol; ++other)
          {
            unsigned int other_length = huffbench_observed.clen[other];
            if (other_length &&
                ((length >= other_length &&
                  (code >> (length - other_length)) == huffbench_observed.code[other]) ||
                 (length < other_length &&
                  (huffbench_observed.code[other] >> (other_length - length)) == code)))
              return 0;
          }
    }
  if (!actual_bits || actual_bits != optimal_bits ||
      huffbench_observed.length != (actual_bits + 7) / 8)
    return 0;
  /* Independently validate every encoded bit against the actual input/codebook.
   * No DUT decoder or editable helper is used by this oracle. */
  for (size_t i = 0; i < HUFFBENCH_TEST_SIZE; ++i)
    {
      unsigned int symbol = orig_data[i];
      unsigned int length = huffbench_observed.clen[symbol];
      for (unsigned int bit = length; bit > 0; --bit, ++position)
        if (((huffbench_observed.bytes[position / 8] >> (7 - position % 8)) & 1u) !=
            ((huffbench_observed.code[symbol] >> (bit - 1)) & 1u))
          return 0;
    }
  for (; position < huffbench_observed.length * 8; ++position)
    if ((huffbench_observed.bytes[position / 8] >> (7 - position % 8)) & 1u)
      return 0;
  return 1;
}
