/*
 * Level 1 of 4: naive OpenMP.
 *
 * The version you write first. Stick "#pragma omp parallel for" on the outer
 * loop, confirm the answer is still right, move on.
 *
 * What is deliberately missing, because later levels add it back:
 *   - no "restrict", so the compiler must assume vx and sxx might overlap and
 *     cannot keep anything in registers across a store
 *   - no "omp simd", and with the aliasing above it will not auto-vectorise
 *   - no schedule clause
 *   - row offsets recomputed as i*nx inside the inner loop
 *
 * Build:
 *   gcc -O3 -fopenmp -march=native -fPIC -shared \
 *       -o libelastic_kernels_v1.so elastic_kernels_v1_naive.c
 */

#include <omp.h>

void update_stress(
    float *vx, float *vz,
    float *sxx, float *szz, float *sxz,
    const float *lam,
    const float *lam2mu,
    const float *mu,
    const float *damp,
    float dt, float dx, float dz,
    int nz, int nx,
    int iz0, int iz1,
    int jx0, int jx1
) {
    const float dtx = dt / dx;
    const float dtz = dt / dz;

    /* The caller's bounds can reach the halo; the stencil reads i+1 and j+1. */
    if (iz0 < 1) iz0 = 1;
    if (iz1 > nz - 1) iz1 = nz - 1;

    #pragma omp parallel for
    for (int i = iz0; i < iz1; ++i) {
        for (int j = jx0; j < jx1; ++j) {
            int k = i * nx + j;
            int kz = (i + 1) * nx + j;

            float dvx_dx = vx[k + 1] - vx[k];
            float dvx_dz = vx[kz] - vx[k];
            float dvz_dx = vz[k + 1] - vz[k];
            float dvz_dz = vz[kz] - vz[k];

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
    float *vx, float *vz,
    const float *sxx,
    const float *szz,
    const float *sxz,
    const float *inv_rho,
    const float *damp,
    float dt, float dx, float dz,
    int nz, int nx,
    int iz0, int iz1,
    int jx0, int jx1
) {
    const float dtx = dt / dx;
    const float dtz = dt / dz;

    if (iz0 < 1) iz0 = 1;
    if (iz1 > nz - 1) iz1 = nz - 1;

    #pragma omp parallel for
    for (int i = iz0; i < iz1; ++i) {
        for (int j = jx0; j < jx1; ++j) {
            int k = i * nx + j;
            int kz = (i - 1) * nx + j;

            float dsxx_dx = sxx[k] - sxx[k - 1];
            float dsxz_dz = sxz[k] - sxz[kz];
            float dsxz_dx = sxz[k] - sxz[k - 1];
            float dszz_dz = szz[k] - szz[kz];

            float ir = inv_rho[k];

            float vx_new = vx[k] + ir * (dtx * dsxx_dx + dtz * dsxz_dz);
            float vz_new = vz[k] + ir * (dtx * dsxz_dx + dtz * dszz_dz);

            float d = damp[k];

            vx[k] = vx_new * d;
            vz[k] = vz_new * d;
        }
    }
}
