/*
 * Level 2 of 4: let the compiler vectorise.
 *
 * Four changes over level 1, and the first two matter most:
 *
 *   restrict on every pointer. Without it the compiler has to assume sxx and
 *   vx might be the same memory, so it reloads after every store and gives up
 *   on vectorising. This one keyword is worth more than the rest combined.
 *
 *   #pragma omp simd on the inner loop, which is only useful once restrict has
 *   made vectorisation legal.
 *
 *   Row base offsets hoisted out of the inner loop, so the address is one add
 *   rather than a multiply-add per access.
 *
 *   schedule(static). The work per row is constant, so the default is probably
 *   static anyway, but saying so removes the doubt and makes thread-to-row
 *   mapping repeatable, which matters for NUMA placement across runs.
 *
 * This is the kernel the repo has been using. Treat it as the baseline that
 * levels 3 and 4 have to beat.
 *
 * Build:
 *   gcc -O3 -fopenmp -march=native -fPIC -shared \
 *       -o libelastic_kernels_v2.so elastic_kernels_v2_simd.c
 */

#include <omp.h>

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

    #pragma omp parallel for schedule(static)
    for (int i = iz0; i < iz1; ++i) {
        int row = i * nx;
        int row_below = (i + 1) * nx;

        #pragma omp simd
        for (int j = jx0; j < jx1; ++j) {
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

    #pragma omp parallel for schedule(static)
    for (int i = iz0; i < iz1; ++i) {
        int row = i * nx;
        int row_above = (i - 1) * nx;

        #pragma omp simd
        for (int j = jx0; j < jx1; ++j) {
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
