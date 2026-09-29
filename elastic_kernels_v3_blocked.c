/*
 * Level 3 of 4: tile for cache.
 *
 * Level 2 works one full row at a time. Row i reads row i+1, and on the next
 * iteration row i+1 is the current row. If a row fits in cache, the second
 * read is cheap. If not, every row comes from memory twice.
 *
 * That depends on nx, the model width plus twice the sponge. At nx = 560 a row
 * is 2.2 kB and there is no problem. At nx = 40000 a row is 160 kB, times the
 * nine arrays this kernel reads, and the reuse is lost.
 *
 * So split the grid into BLOCK_ROWS x BLOCK_COLUMNS tiles and give threads
 * tiles instead of rows. Three rows of a tile then fit in L1 at any grid
 * width. collapse(2) merges the two tile loops so every thread gets work.
 *
 * Each kernel reads one array and writes another, and no cell depends on
 * another cell of the same array. So tile order gives the same values as row
 * order.
 *
 * Set block sizes at build time:
 *   -DBLOCK_ROWS=32 -DBLOCK_COLUMNS=512
 *
 * 512 columns is 2 kB per array per row. About eleven array rows are in use
 * inside a tile, so about 22 kB, which fits a 32 kB L1. Try wider until it
 * stops helping. If each rank's tile is small, there may be too few tiles to
 * keep every thread busy, and level 2 will be faster.
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