#ifndef CFU_COUNTER_H
#define CFU_COUNTER_H

#include <stdint.h>

#if defined(__riscv)
static inline uint32_t cfu_counter_read(void) {
    uint32_t value;
    __asm__ volatile("rdcycle %0" : "=r"(value));
    return value;
}
#else
#include <time.h>
static inline uint32_t cfu_counter_read(void) {
    return (uint32_t)clock();
}
#endif

static inline long cfu_counter_elapsed(uint32_t start, uint32_t end) {
    return (long)(uint32_t)(end - start);
}

#endif
