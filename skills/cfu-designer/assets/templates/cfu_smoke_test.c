#include <stdint.h>
#include <stdio.h>
#include "cfu_intrinsics.h"

static int32_t scalar_ref(int32_t a, int32_t b, int32_t cfg) {
    return a + b + cfg;
}

int main(void) {
    const int32_t cfg = 7;
    const int32_t cases[][2] = {
        {1, 2},
        {-4, 9},
        {123, -55},
        {0x1000, 0x20},
    };

    cfu_enable();
    cfu_config(7);

    for(unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int32_t got = cfu_compute(cases[i][0], cases[i][1]);
        int32_t exp = scalar_ref(cases[i][0], cases[i][1], cfg);
        if(got != exp) {
            printf("CFU mismatch case=%u got=%ld exp=%ld\n", i, (long)got, (long)exp);
            return 1;
        }
    }

    printf("CFU smoke PASS\n");
    return 0;
}
