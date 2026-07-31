#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdint.h>
#include <stdio.h>
#endif

enum {
    BYTES = 128,
    REPEAT = 8
};

static volatile uint8_t x[BYTES] __attribute__((aligned(32)));
static volatile uint8_t w[BYTES] __attribute__((aligned(32)));
static volatile uint16_t product[BYTES] __attribute__((aligned(32)));
volatile uint32_t cfu_agent_example_sink = 0;

static void init_data(void) {
    for (int i = 0; i < BYTES; ++i) {
        x[i] = (uint8_t)((i * 17 + 3) & 0xff);
        w[i] = (uint8_t)((i * 29 + 91) & 0xff);
        product[i] = 0;
    }
}

__attribute__((noinline))
static uint32_t scalar_fused(const volatile uint8_t *a, const volatile uint8_t *b, int bytes) {
    uint32_t sum = 0;
    for (int i = 0; i < bytes; ++i) {
        sum += (uint32_t)a[i] * (uint32_t)b[i];
    }
    return sum / (uint32_t)bytes;
}

__attribute__((noinline))
static uint32_t scalar_two_pass(const volatile uint8_t *a, const volatile uint8_t *b, volatile uint16_t *tmp, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        tmp[i] = (uint16_t)((uint32_t)a[i] * (uint32_t)b[i]);
    }

    uint32_t sum = 0;
    for (int i = 0; i < bytes; ++i) {
        sum += tmp[i];
    }
    return sum / (uint32_t)bytes;
}

int main(void) {
    init_data();

    uint32_t fused = 0;
    uint32_t two_pass = 0;
    for (int r = 0; r < REPEAT; ++r) {
        fused += scalar_fused(x, w, BYTES);
        two_pass += scalar_two_pass(x, w, product, BYTES);
    }

    if (fused != two_pass) {
        printf("CFU_AGENT_FAIL fused=%u two_pass=%u\n", fused, two_pass);
        return 1;
    }

    cfu_agent_example_sink = fused;
    printf("CFU_AGENT_PASS fused=%u repeat=%d bytes=%d\n", fused, REPEAT, BYTES);
    return 0;
}
