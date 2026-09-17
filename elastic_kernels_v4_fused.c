/*
 * Level 4 of 4: squeeze the loads, and find out whether it was worth it.
 *
 * Two micro-optimisations on top of level 3.
 *
 * Row-pair fusion. In update_stress, row i reads row i+1, and row i+1 reads
 * row i+2. Process the rows two at a time and the shared middle row is loaded
 * once instead of twice: three row loads of vx and vz per pair rather than
 * four. update_velocity has the same overlap one row up.
 *
 * Software prefetch. The hardware prefetcher follows a sequential walk along a
 * row without help. What it handles less well is the jump to the next row,
 * which in a tiled loop is a stride of nx floats to an address the walk has
 * not touched. So issue a prefetch for the next pair's rows while the current
 * pair is being computed.
 *
 * Be sceptical of this level. It is where you stop removing obvious waste and
 * start fighting for single-digit percentages, and both changes can backfire:
 * the fused body needs more live registers and may spill, and prefetch
 * instructions cost issue slots whether or not they help. On a memory-bound
 * kernel already running at bandwidth, neither can do anything at all. The
 * honest outcome of this level might be "no faster than level 3, and harder to
 * read", which is a perfectly good result to report as long as you measured it
 * rather than assumed it.
 *
 * Arithmetic is unchanged, so the output is still bit-for-bit identical to
 * every other level.
 *
 * Build:
 *   gcc -O3 -fopenmp -march=native -fPIC -shared \
 *       -o libelastic_kernels_v4.so elastic_kernels_v4_fused.c
 */

#include <omp.h>

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 32
#endif

#ifndef BLOCK_COLUMNS
#define BLOCK_COLUMNS 512
#endif

/* Floats per 64-byte cache line. One prefetch per line is enough. */
#define FLOATS_PER_LINE 16

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

            int i = i_block;

            /* Two rows at a time, sharing the row between them. */
            for (; i + 1 < i_end; i += 2) {
                const int row_top = i * nx;
                const int row_mid = (i + 1) * nx;
                const int row_bot = (i + 2) * nx;

                if (i + 3 < i_end) {
                    const int row_next = (i + 3) * nx;
                    for (int j = j_block; j < j_end; j += FLOATS_PER_LINE) {
                        __builtin_prefetch(&vx[row_next + j], 0, 1);
                        __builtin_prefetch(&vz[row_next + j], 0, 1);
                    }
                }

                #pragma omp simd
                for (int j = j_block; j < j_end; ++j) {
                    const int k_top = row_top + j;
                    const int k_mid = row_mid + j;

                    /* Loaded once, used by both rows. */
                    const float vx_mid = vx[k_mid];
                    const float vz_mid = vz[k_mid];

                    float dvx_dx = vx[k_top + 1] - vx[k_top];
                    float dvx_dz = vx_mid - vx[k_top];
                    float dvz_dx = vz[k_top + 1] - vz[k_top];
                    float dvz_dz = vz_mid - vz[k_top];

                    float sxx_new = sxx[k_top] + lam2mu[k_top] * dtx * dvx_dx
                                  + lam[k_top] * dtz * dvz_dz;
                    float szz_new = szz[k_top] + lam[k_top] * dtx * dvx_dx
                                  + lam2mu[k_top] * dtz * dvz_dz;
                    float sxz_new = sxz[k_top]
                                  + mu[k_top] * (dtz * dvx_dz + dtx * dvz_dx);

                    float d_top = damp[k_top];

                    sxx[k_top] = sxx_new * d_top;
                    szz[k_top] = szz_new * d_top;
                    sxz[k_top] = sxz_new * d_top;

                    dvx_dx = vx[k_mid + 1] - vx_mid;
                    dvx_dz = vx[row_bot + j] - vx_mid;
                    dvz_dx = vz[k_mid + 1] - vz_mid;
                    dvz_dz = vz[row_bot + j] - vz_mid;

                    sxx_new = sxx[k_mid] + lam2mu[k_mid] * dtx * dvx_dx
                            + lam[k_mid] * dtz * dvz_dz;
                    szz_new = szz[k_mid] + lam[k_mid] * dtx * dvx_dx
                            + lam2mu[k_mid] * dtz * dvz_dz;
                    sxz_new = sxz[k_mid]
                            + mu[k_mid] * (dtz * dvx_dz + dtx * dvz_dx);

                    float d_mid = damp[k_mid];

                    sxx[k_mid] = sxx_new * d_mid;
                    szz[k_mid] = szz_new * d_mid;
                    sxz[k_mid] = sxz_new * d_mid;
                }
            }

            /* Odd tile height leaves one row over. */
            for (; i < i_end; ++i) {
                const int row = i * nx;
                const int row_below = (i + 1) * nx;

                #pragma omp simd
                for (int j = j_block; j < j_end; ++j) {
                    int k = row + j;

                    float dvx_dx = vx[k + 1] - vx[k];
                    float dvx_dz = vx[row_below + j] - vx[k];
                    float dvz_dx = vz[k + 1] - vz[k];
                    float dvz_dz = vz[row_below + j] - vz[k];

                    float sxx_new = sxx[k] + lam2mu[k] * dtx * dvx_dx
                                  + lam[k] * dtz * dvz_dz;
                    float szz_new = szz[k] + lam[k] * dtx * dvx_dx
                                  + lam2mu[k] * dtz * dvz_dz;
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

            int i = i_block;

            /* Rows i and i+1 both read row i, so the pair shares it. */
            for (; i + 1 < i_end; i += 2) {
                const int row_above = (i - 1) * nx;
                const int row_top = i * nx;
                const int row_mid = (i + 1) * nx;

                if (i + 3 < i_end) {
                    const int row_next = (i + 3) * nx;
                    for (int j = j_block; j < j_end; j += FLOATS_PER_LINE) {
                        __builtin_prefetch(&sxz[row_next + j], 0, 1);
                        __builtin_prefetch(&szz[row_next + j], 0, 1);
                    }
                }

                #pragma omp simd
                for (int j = j_block; j < j_end; ++j) {
                    const int k_top = row_top + j;
                    const int k_mid = row_mid + j;

                    /* Loaded once, used by both rows. */
                    const float sxz_top = sxz[k_top];
                    const float szz_top = szz[k_top];

                    float dsxx_dx = sxx[k_top] - sxx[k_top - 1];
                    float dsxz_dz = sxz_top - sxz[row_above + j];
                    float dsxz_dx = sxz_top - sxz[k_top - 1];
                    float dszz_dz = szz_top - szz[row_above + j];

                    float ir = inv_rho[k_top];

                    float vx_new = vx[k_top] + ir * (dtx * dsxx_dx + dtz * dsxz_dz);
                    float vz_new = vz[k_top] + ir * (dtx * dsxz_dx + dtz * dszz_dz);

                    float d_top = damp[k_top];

                    vx[k_top] = vx_new * d_top;
                    vz[k_top] = vz_new * d_top;

                    dsxx_dx = sxx[k_mid] - sxx[k_mid - 1];
                    dsxz_dz = sxz[k_mid] - sxz_top;
                    dsxz_dx = sxz[k_mid] - sxz[k_mid - 1];
                    dszz_dz = szz[k_mid] - szz_top;

                    ir = inv_rho[k_mid];

                    vx_new = vx[k_mid] + ir * (dtx * dsxx_dx + dtz * dsxz_dz);
                    vz_new = vz[k_mid] + ir * (dtx * dsxz_dx + dtz * dszz_dz);

                    float d_mid = damp[k_mid];

                    vx[k_mid] = vx_new * d_mid;
                    vz[k_mid] = vz_new * d_mid;
                }
            }

            for (; i < i_end; ++i) {
                const int row = i * nx;
                const int row_above = (i - 1) * nx;

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
