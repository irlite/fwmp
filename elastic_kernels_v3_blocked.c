/*
 * Level 3 of 4: tile for cache.
 *
 * Level 2 walks a whole row at a time. Row i reads the row below it, and one
 * iteration later that row becomes row i. If a row fits in cache, the second
 * read is free. If it does not, every row gets fetched from memory twice and
 * the kernel runs at memory bandwidth for no good reason.
 *
 * Whether that happens depends on nx, which is the model width plus twice the
 * sponge. At nx = 560 a row is 2.2 kB and there is nothing to fix. At nx =
 * 40000 it is 160 kB, times the nine arrays this kernel touches, and the reuse
 * is gone.
 *
 * So: split the grid into BLOCK_ROWS x BLOCK_COLUMNS tiles and hand out tiles
 * instead of rows. Each thread then works inside a window whose three-row
 * footprint fits L1 no matter how wide the grid is. collapse(2) flattens the
 * two tile loops so there are enough chunks to keep every thread busy.
 *
 * Both kernels read one array and write another, with no cell depending on
 * another cell of the same array, so visiting cells in tile order instead of
 * row order computes exactly the same values.
 *
 * Tune the block sizes at build time:
 *   -DBLOCK_ROWS=32 -DBLOCK_COLUMNS=512
 *
 * 512 columns is 2 kB per array per row. With roughly eleven array-rows live
 * inside a tile that is about 22 kB, which fits a 32 kB L1. Widen it until it
 * stops helping. If your tile per rank is small, watch out for the other
 * failure mode: too few tiles to feed the threads, and level 2 wins.
 *
 * Build:
 *   gcc -O3 -fopenmp -march=native -fPIC -shared \
 *       -o libelastic_kernels_v3.so elastic_kernels_v3_blocked.c
 */

#include <omp.h>

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 32
#endif

#ifndef BLOCK_COLUMNS
#define BLOCK_COLUMNS 512
#endif

static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

void update_stress(
    float * restrict vx, float * restrict vz,
    float * restrict sxx, float * restrict szz, float * restrict sxz,
    const float * restrict lam,
    const float * restrict lam2mu,
    const float * restrict mu,
    const float * restrict damp,
    float dt, float dx, float dz,
    int nz, int nx,
    int iz0, int iz1,
    int jx0, int jx1
) {
    const float dtx = dt / dx;
    const float dtz = dt / dz;

    if (iz0 < 1) iz0 = 1;
    if (iz1 > nz - 1) iz1 = nz - 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i_block = iz0; i_block < iz1; i_block += BLOCK_ROWS) {
        for (int j_block = jx0; j_block < jx1; j_block += BLOCK_COLUMNS) {

            const int i_end = min_int(i_block + BLOCK_ROWS, iz1);
            const int j_end = min_int(j_block + BLOCK_COLUMNS, jx1);

            for (int i = i_block; i < i_end; ++i) {
                int row = i * nx;
                int row_below = (i + 1) * nx;

                #pragma omp simd
                for (int j = j_block; j < j_end; ++j) {
                    int k = row + j;

                    float dvx_dx = vx[k + 1] - vx[k];
                    float dvx_dz = vx[row_below + j] - vx[k];
                    float dvz_dx = vz[k + 1] - vz[k];
                    float dvz_dz = vz[row_below + j] - vz[k];

                    float sxx_new = sxx[k] + lam2mu[k] * dtx * dvx_dx + lam[k] * dtz * dvz_dz;
                    float szz_new = szz[k] + lam[k] * dtx * dvx_dx + lam2mu[k] * dtz * dvz_dz;
                    float sxz_new = sxz[k] + mu[k] * (dtz * dvx_dz + dtx * dvz_dx);

                    float d = damp[k];

                    sxx[k] = sxx_new * d;
                    szz[k] = szz_new * d;
                    sxz[k] = sxz_new * d;
                }
            }
        }
    }
}

void update_velocity(
    float * restrict vx, float * restrict vz,
    const float * restrict sxx,
    const float * restrict szz,
    const float * restrict sxz,
    const float * restrict inv_rho,
    const float * restrict damp,
    float dt, float dx, float dz,
    int nz, int nx,
    int iz0, int iz1,
    int jx0, int jx1
) {
    const float dtx = dt / dx;
    const float dtz = dt / dz;

    if (iz0 < 1) iz0 = 1;
    if (iz1 > nz - 1) iz1 = nz - 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i_block = iz0; i_block < iz1; i_block += BLOCK_ROWS) {
        for (int j_block = jx0; j_block < jx1; j_block += BLOCK_COLUMNS) {

            const int i_end = min_int(i_block + BLOCK_ROWS, iz1);
            const int j_end = min_int(j_block + BLOCK_COLUMNS, jx1);

            for (int i = i_block; i < i_end; ++i) {
                int row = i * nx;
                int row_above = (i - 1) * nx;

                #pragma omp simd
                for (int j = j_block; j < j_end; ++j) {
                    int k = row + j;

                    float dsxx_dx = sxx[k] - sxx[k - 1];
                    float dsxz_dz = sxz[k] - sxz[row_above + j];
                    float dsxz_dx = sxz[k] - sxz[k - 1];
                    float dszz_dz = szz[k] - szz[row_above + j];

                    float ir = inv_rho[k];

                    float vx_new = vx[k] + ir * (dtx * dsxx_dx + dtz * dsxz_dz);
                    float vz_new = vz[k] + ir * (dtx * dsxz_dx + dtz * dszz_dz);

                    float d = damp[k];

                    vx[k] = vx_new * d;
                    vz[k] = vz_new * d;
                }
            }
        }
    }
}
