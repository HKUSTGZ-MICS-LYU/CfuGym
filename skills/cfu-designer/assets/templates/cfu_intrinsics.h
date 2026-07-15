#ifndef STREAMING_CFU_INTRINSICS_H
#define STREAMING_CFU_INTRINSICS_H

#include <stdint.h>

#define cfu_enable() do { \
    __asm__ volatile( \
        "li t1, 0x80000000\n\t" \
        "csrs 0xBC0, t1" \
        ::: "t1" \
    ); \
} while(0)

#define cfu_fence() __asm__ volatile("fence rw, rw" ::: "memory")

// custom0, func3=2. This template passes config in raw rs1 bits.
#define cfu_config(cfg) do { \
    __asm__ volatile( \
        ".word (0x0B | (0 << 7) | ((%0) << 15) | (0 << 20) | (0x2 << 12) | (0 << 25))" \
        :: "i"(cfg) \
    ); \
} while(0)

// custom0, func3=4. rs1 carries the address value; raw rs2 carries destination CFU register id.
#define cfu_load(dst_reg, addr) do { \
    register uintptr_t _addr_reg asm("a5") = (uintptr_t)(addr); \
    __asm__ volatile( \
        ".word (0x0B | (0 << 7) | (15 << 15) | ((%1) << 20) | (0x4 << 12) | (0 << 25))" \
        : "+r"(_addr_reg) : "i"(dst_reg) : "memory" \
    ); \
} while(0)

// custom0, func3=1. rs1/rs2 values are normal scalar operands in this template.
static inline int32_t cfu_compute(int32_t a, int32_t b) {
    register int32_t _a asm("a5") = a;
    register int32_t _b asm("a6") = b;
    register int32_t _out asm("t0");
    __asm__ volatile(
        ".word (0x0B | (5 << 7) | (15 << 15) | (16 << 20) | (0x1 << 12) | (0 << 25))"
        : "=r"(_out)
        : "r"(_a), "r"(_b)
    );
    return _out;
}

#endif
