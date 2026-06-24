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
    if (jx0 < 1) jx0 = 1;
    if (jx1 > nx - 1) jx1 = nx - 1;

    #pragma omp parallel for schedule(static)
    for (int i = iz0; i < iz1; ++i) {
        int row = i * nx;
        int rowp = (i + 1) * nx;

        #pragma omp simd
        for (int j = jx0; j < jx1; ++j) {
            int k = row + j;

            float dvx_dx = vx[k + 1] - vx[k];
            float dvx_dz = vx[rowp + j] - vx[k];
            float dvz_dx = vz[k + 1] - vz[k];
            float dvz_dz = vz[rowp + j] - vz[k];

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
    if (jx0 < 1) jx0 = 1;
    if (jx1 > nx - 1) jx1 = nx - 1;

    #pragma omp parallel for schedule(static)
    for (int i = iz0; i < iz1; ++i) {
        int row = i * nx;
        int rowm = (i - 1) * nx;

        #pragma omp simd
        for (int j = jx0; j < jx1; ++j) {
            int k = row + j;

            float dsxx_dx = sxx[k] - sxx[k - 1];
            float dsxz_dz = sxz[k] - sxz[rowm + j];
            float dsxz_dx = sxz[k] - sxz[k - 1];
            float dszz_dz = szz[k] - szz[rowm + j];

            float ir = inv_rho[k];

            float vx_new = vx[k] + ir * (dtx * dsxx_dx + dtz * dsxz_dz);
            float vz_new = vz[k] + ir * (dtx * dsxz_dx + dtz * dszz_dz);

            float d = damp[k];

            vx[k] = vx_new * d;
            vz[k] = vz_new * d;
        }
    }
}
