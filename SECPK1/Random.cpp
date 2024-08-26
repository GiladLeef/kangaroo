#include "Random.h"

#define RK_STATE_LEN 624
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

typedef struct rk_state_ {
    unsigned long key[RK_STATE_LEN];
    int pos;
} rk_state;

rk_state localState;

void rk_seed(unsigned long seed, rk_state *state) {
    int pos;
    seed &= 0xffffffffUL;

    // Initialize the state array with the seed
    for (pos = 0; pos < RK_STATE_LEN; ++pos) {
        state->key[pos] = seed;
        seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;
    }

    state->pos = RK_STATE_LEN;
}

inline unsigned long rk_random(rk_state *state) {
    unsigned long y;
    unsigned long *key = state->key;

    if (state->pos == RK_STATE_LEN) {
        int i;
        unsigned long y;

        // Unrolling loop to speed up key generation
        for (i = 0; i < N - M; ++i) {
            y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
            key[i] = key[i + M] ^ (y >> 1) ^ ((y & 1) ? MATRIX_A : 0);
        }
        for (; i < N - 1; ++i) {
            y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
            key[i] = key[i + (M - N)] ^ (y >> 1) ^ ((y & 1) ? MATRIX_A : 0);
        }
        y = (key[N - 1] & UPPER_MASK) | (key[0] & LOWER_MASK);
        key[N - 1] = key[M - 1] ^ (y >> 1) ^ ((y & 1) ? MATRIX_A : 0);

        state->pos = 0;
    }

    y = key[state->pos++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

inline double rk_double(rk_state *state) {
    /* More efficient calculation for the double value */
    unsigned long a = rk_random(state) >> 5;
    unsigned long b = rk_random(state) >> 6;
    // Instead of multiplication and division, use precomputed constants
    return (a * 0x1.0p-26 + b * 0x1.0p-53);
}

void rseed(unsigned long seed) {
    rk_seed(seed, &localState);
}

unsigned long rndl() {
    return rk_random(&localState);
}

double rnd() {
    return rk_double(&localState);
}
