/* Host-owned Huffbench stage interface; copied identically to all builds.
 * The original fixture is 500 bytes. Codes use explicit RV32-width words.
 * Alternative optimal prefix codes are allowed; metadata and encoded bytes
 * must describe the actual input. Success is 1, failure/unsupported is 0.
 */
#ifndef EMBENCH_HUFFBENCH_H
#define EMBENCH_HUFFBENCH_H
#include <stddef.h>
#include <stdint.h>
#define HUFFBENCH_TEST_SIZE 500
#define HUFFBENCH_CAPACITY (HUFFBENCH_TEST_SIZE + 1)
struct huffbench_encoded {
  size_t length;
  uint32_t code[256];
  unsigned char clen[256];
  unsigned char bytes[HUFFBENCH_CAPACITY];
};
int huffbench_encode(const unsigned char *data, size_t data_len,
                    struct huffbench_encoded *encoded);
int huffbench_decode(const struct huffbench_encoded *encoded,
                    unsigned char *data, size_t data_len);
#endif
