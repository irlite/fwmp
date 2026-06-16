#include <omp.h>

void update_stress(
    float *vx, float *vz,
    float *sxx, float *szz, float *sxz,
    const float *lam, const float *lam2mu, const float *mu, const float *damp,
    float dt, float dx, float dz,
    int nz, int nx,
    int jx0, int jx1
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < nz - 1; ++i) {
        for (int j = jx0; j < jx1; ++j) {
            int k = i * nx + j;
            float dvx_dx = (vx[i * nx + (j + 1)] - vx[k]) / dx;
            float dvx_dz = (vx[(i + 1) * nx + j] - vx[k]) / dz;
            float dvz_dx = (vz[i * nx + (j + 1)] - vz[k]) / dx;
            float dvz_dz = (vz[(i + 1) * nx + j] - vz[k]) / dz;
            sxx[k] += dt * (lam2mu[k] * dvx_dx + lam[k] * dvz_dz);
            szz[k] += dt * (lam[k] * dvx_dx + lam2mu[k] * dvz_dz);
            sxz[k] += dt * mu[k] * (dvx_dz + dvz_dx);
            float d = damp[k];
            sxx[k] *= d;
            szz[k] *= d;
            sxz[k] *= d;
        }
    }
}

void update_velocity(
    float *vx, float *vz,
    const float *sxx, const float *szz, const float *sxz,
    const float *inv_rho, const float *damp,
    float dt, float dx, float dz,
    int nz, int nx,
    int jx0, int jx1
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < nz - 1; ++i) {
        for (int j = jx0; j < jx1; ++j) {
            int k = i * nx + j;
            float dsxx_dx = (sxx[k] - sxx[i * nx + (j - 1)]) / dx;
            float dsxz_dz = (sxz[k] - sxz[(i - 1) * nx + j]) / dz;
            float dsxz_dx = (sxz[k] - sxz[i * nx + (j - 1)]) / dx;
            float dszz_dz = (szz[k] - szz[(i - 1) * nx + j]) / dz;
            vx[k] += dt * inv_rho[k] * (dsxx_dx + dsxz_dz);
            vz[k] += dt * inv_rho[k] * (dsxz_dx + dszz_dz);
            float d = damp[k];
            vx[k] *= d;
            vz[k] *= d;
        }
    }
}
