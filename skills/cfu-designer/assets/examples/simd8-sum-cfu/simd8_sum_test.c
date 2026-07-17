#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdint.h>
#include <stdio.h>
#endif

#include "cfu_counter.h"

volatile int32_t g_simd8_sum_scalar = 0;
volatile int32_t g_simd8_sum_cfu = 0;
volatile uint32_t g_simd8_sum_scalar_cycles = 0;
volatile uint32_t g_simd8_sum_cfu_cycles = 0;
volatile uint32_t g_simd8_sum_speedup_x100 = 0;
volatile int32_t g_simd8_sum_sink = 0;

static inline void cfu_enable(void) {
    __asm__ volatile(
        "li t1, 0x80000000\n\t"
        "csrs 0xBC0, t1"
        ::: "t1"
    );
}

// custom0, func3=0: a0 = sum_i8x4(a0), rs2 is x0.
static inline int32_t cfu_sum_i8x4(int32_t packed) {
    register int32_t inout asm("a0") = packed;
    __asm__ volatile(
        ".word (0x0B | (10 << 7) | (10 << 15) | (0 << 20) | (0x0 << 12) | (0 << 25))"
        : "+r"(inout)
    );
    return inout;
}

static uint32_t pack_i8x4_values(int8_t x0, int8_t x1, int8_t x2, int8_t x3) {
    return ((uint32_t)(uint8_t)x0 << 0) |
           ((uint32_t)(uint8_t)x1 << 8) |
           ((uint32_t)(uint8_t)x2 << 16) |
           ((uint32_t)(uint8_t)x3 << 24);
}

__attribute__((noinline))
static int32_t scalar_sum_i8_words(const volatile uint32_t *x, int words) {
    int32_t sum = 0;
    for(int i = 0; i < words; ++i) {
        uint32_t packed = x[i];
        sum += (int32_t)(int8_t)(packed >> 0);
        sum += (int32_t)(int8_t)(packed >> 8);
        sum += (int32_t)(int8_t)(packed >> 16);
        sum += (int32_t)(int8_t)(packed >> 24);
    }
    return sum;
}

__attribute__((noinline))
static int32_t cfu_sum_i8_words(const volatile uint32_t *x, int words) {
    int32_t sum = 0;
    for(int i = 0; i < words; ++i) {
        sum += cfu_sum_i8x4((int32_t)x[i]);
    }
    return sum;
}

int main(void) {
    enum { WORDS = 256, REPEAT = 128 };
    static volatile uint32_t data[WORDS];

    for(int i = 0; i < WORDS; ++i) {
        int base = i * 4;
        data[i] = pack_i8x4_values(
            (int8_t)(((base + 0) * 17 + 3) % 127 - 63),
            (int8_t)(((base + 1) * 17 + 3) % 127 - 63),
            (int8_t)(((base + 2) * 17 + 3) % 127 - 63),
            (int8_t)(((base + 3) * 17 + 3) % 127 - 63));
    }

    cfu_enable();

    int32_t scalar = scalar_sum_i8_words(data, WORDS);
    int32_t accel = cfu_sum_i8_words(data, WORDS);
    g_simd8_sum_scalar = scalar;
    g_simd8_sum_cfu = accel;
    if(scalar != accel) {
        return 1;
    }

    g_simd8_sum_sink = scalar_sum_i8_words(data, WORDS);
    g_simd8_sum_sink += cfu_sum_i8_words(data, WORDS);

    uint32_t start_time = cfu_counter_read();
    int32_t scalar_acc = 0;
    for(int r = 0; r < REPEAT; ++r) {
        scalar_acc += scalar_sum_i8_words(data, WORDS);
    }
    long scalar_time = cfu_counter_elapsed(start_time, cfu_counter_read());

    start_time = cfu_counter_read();
    int32_t cfu_acc = 0;
    for(int r = 0; r < REPEAT; ++r) {
        cfu_acc += cfu_sum_i8_words(data, WORDS);
    }
    long cfu_time = cfu_counter_elapsed(start_time, cfu_counter_read());

    if(scalar_acc != cfu_acc) {
        return 2;
    }

    long speedup_x100 = cfu_time > 0 ? (scalar_time * 100L) / cfu_time : 0L;
    g_simd8_sum_sink = scalar_acc + cfu_acc;
    g_simd8_sum_scalar_cycles = (uint32_t)scalar_time;
    g_simd8_sum_cfu_cycles = (uint32_t)cfu_time;
    g_simd8_sum_speedup_x100 = (uint32_t)speedup_x100;

    printf("SIMD8_SUM PASS scalar=%d cfu=%d\n", (int)scalar, (int)accel);
    printf("SIMD8_SUM_PROFILE scalar=%d cfu=%d speedup_x100=%d\n",
           (int)scalar_time, (int)cfu_time, (int)speedup_x100);

    return 0;
}
