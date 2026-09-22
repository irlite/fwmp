#include <omp.h>
#include <stdlib.h>

static inline void stress_row(
    const float * restrict vx,
    const float * restrict vz,
    float * restrict sxx,
    float * restrict szz,
    float * restrict sxz,
    const float * restrict lam,
    const float * restrict lam2mu,
    const float * restrict mu,
    const float * restrict damp,
    float dtx,
    float dtz,
    int nx,
    int i,
    int jx0,
    int jx1
) {
    const int row = i * nx;
    const int rowp = row + nx;

    #pragma omp simd
    for (int j = jx0; j < jx1; ++j) {
        const int k = row + j;
        const float dvx_dx = vx[k + 1] - vx[k];
        const float dvx_dz = vx[rowp + j] - vx[k];
        const float dvz_dx = vz[k + 1] - vz[k];
        const float dvz_dz = vz[rowp + j] - vz[k];
        const float sxx_new = sxx[k] + lam2mu[k] * dtx * dvx_dx + lam[k] * dtz * dvz_dz;
        const float szz_new = szz[k] + lam[k] * dtx * dvx_dx + lam2mu[k] * dtz * dvz_dz;
        const float sxz_new = sxz[k] + mu[k] * (dtz * dvx_dz + dtx * dvz_dx);
        const float d = damp[k];
        sxx[k] = sxx_new * d;
        szz[k] = szz_new * d;
        sxz[k] = sxz_new * d;
    }
}

static inline void stress_point(
    const float * restrict vx,
    const float * restrict vz,
    float * restrict sxx,
    float * restrict szz,
    float * restrict sxz,
    const float * restrict lam,
    const float * restrict lam2mu,
    const float * restrict mu,
    const float * restrict damp,
    float dtx,
    float dtz,
    int nx,
    int i,
    int j
) {
    const int k = i * nx + j;
    const float dvx_dx = vx[k + 1] - vx[k];
    const float dvx_dz = vx[k + nx] - vx[k];
    const float dvz_dx = vz[k + 1] - vz[k];
    const float dvz_dz = vz[k + nx] - vz[k];
    const float sxx_new = sxx[k] + lam2mu[k] * dtx * dvx_dx + lam[k] * dtz * dvz_dz;
    const float szz_new = szz[k] + lam[k] * dtx * dvx_dx + lam2mu[k] * dtz * dvz_dz;
    const float sxz_new = sxz[k] + mu[k] * (dtz * dvx_dz + dtx * dvz_dx);
    const float d = damp[k];
    sxx[k] = sxx_new * d;
    szz[k] = szz_new * d;
    sxz[k] = sxz_new * d;
}

static inline void velocity_row(
    float * restrict vx,
    float * restrict vz,
    const float * restrict sxx,
    const float * restrict szz,
    const float * restrict sxz,
    const float * restrict inv_rho,
    const float * restrict damp,
    float dtx,
    float dtz,
    int nx,
    int i,
    int jx0,
    int jx1
) {
    const int row = i * nx;
    const int rowm = row - nx;

    #pragma omp simd
    for (int j = jx0; j < jx1; ++j) {
        const int k = row + j;
        const float dsxx_dx = sxx[k] - sxx[k - 1];
        const float dsxz_dz = sxz[k] - sxz[rowm + j];
        const float dsxz_dx = sxz[k] - sxz[k - 1];
        const float dszz_dz = szz[k] - szz[rowm + j];
        const float ir = inv_rho[k];
        const float vx_new = vx[k] + ir * (dtx * dsxx_dx + dtz * dsxz_dz);
        const float vz_new = vz[k] + ir * (dtx * dsxz_dx + dtz * dszz_dz);
        const float d = damp[k];
        vx[k] = vx_new * d;
        vz[k] = vz_new * d;
    }
}

static inline void velocity_point(
    float * restrict vx,
    float * restrict vz,
    const float * restrict sxx,
    const float * restrict szz,
    const float * restrict sxz,
    const float * restrict inv_rho,
    const float * restrict damp,
    float dtx,
    float dtz,
    int nx,
    int i,
    int j
) {
    const int k = i * nx + j;
    const float dsxx_dx = sxx[k] - sxx[k - 1];
    const float dsxz_dz = sxz[k] - sxz[k - nx];
    const float dsxz_dx = sxz[k] - sxz[k - 1];
    const float dszz_dz = szz[k] - szz[k - nx];
    const float ir = inv_rho[k];
    const float vx_new = vx[k] + ir * (dtx * dsxx_dx + dtz * dsxz_dz);
    const float vz_new = vz[k] + ir * (dtx * dsxz_dx + dtz * dszz_dz);
    const float d = damp[k];
    vx[k] = vx_new * d;
    vz[k] = vz_new * d;
}

static int get_tile_rows(void) {
    const char *value = getenv("FWMP_TILE_ROWS");

    if (value != NULL) {
        const int requested = atoi(value);
        if (requested > 0) {
            return requested;
        }
    }

    int rows = 4 * omp_get_max_threads();

    if (rows < 16) {
        rows = 16;
    }

    if (rows > 256) {
        rows = 256;
    }

    return rows;
}

void update_stress_edges(
    const float * restrict vx,
    const float * restrict vz,
    float * restrict sxx,
    float * restrict szz,
    float * restrict sxz,
    const float * restrict lam,
    const float * restrict lam2mu,
    const float * restrict mu,
    const float * restrict damp,
    float dt,
    float dx,
    float dz,
    int nz,
    int nx,
    int iz0,
    int iz1,
    int jx0,
    int jx1
) {
    if (iz0 < 1) {
        iz0 = 1;
    }

    if (iz1 > nz - 1) {
        iz1 = nz - 1;
    }

    if (jx0 < 1) {
        jx0 = 1;
    }

    if (jx1 > nx - 1) {
        jx1 = nx - 1;
    }

    if (iz0 >= iz1 || jx0 >= jx1) {
        return;
    }

    const float dtx = dt / dx;
    const float dtz = dt / dz;
    const int last_i = iz1 - 1;
    const int last_j = jx1 - 1;

    #pragma omp parallel
    {
        #pragma omp for simd schedule(static)
        for (int j = jx0; j < jx1; ++j) {
            const int k = last_i * nx + j;
            const float dvx_dx = vx[k + 1] - vx[k];
            const float dvx_dz = vx[k + nx] - vx[k];
            const float dvz_dx = vz[k + 1] - vz[k];
            const float dvz_dz = vz[k + nx] - vz[k];
            const float sxx_new = sxx[k] + lam2mu[k] * dtx * dvx_dx + lam[k] * dtz * dvz_dz;
            const float szz_new = szz[k] + lam[k] * dtx * dvx_dx + lam2mu[k] * dtz * dvz_dz;
            const float sxz_new = sxz[k] + mu[k] * (dtz * dvx_dz + dtx * dvz_dx);
            const float d = damp[k];
            sxx[k] = sxx_new * d;
            szz[k] = szz_new * d;
            sxz[k] = sxz_new * d;
        }

        #pragma omp for schedule(static)
        for (int i = iz0; i < last_i; ++i) {
            stress_point(
                vx,
                vz,
                sxx,
                szz,
                sxz,
                lam,
                lam2mu,
                mu,
                damp,
                dtx,
                dtz,
                nx,
                i,
                last_j
            );
        }
    }
}

void update_stress_velocity_interior(
    float * restrict vx,
    float * restrict vz,
    float * restrict sxx,
    float * restrict szz,
    float * restrict sxz,
    const float * restrict lam,
    const float * restrict lam2mu,
    const float * restrict mu,
    const float * restrict inv_rho,
    const float * restrict damp,
    float dt,
    float dx,
    float dz,
    int nz,
    int nx,
    int iz0,
    int iz1,
    int jx0,
    int jx1
) {
    if (iz0 < 1) {
        iz0 = 1;
    }

    if (iz1 > nz - 1) {
        iz1 = nz - 1;
    }

    if (jx0 < 1) {
        jx0 = 1;
    }

    if (jx1 > nx - 1) {
        jx1 = nx - 1;
    }

    if (iz0 >= iz1 || jx0 >= jx1) {
        return;
    }

    const float dtx = dt / dx;
    const float dtz = dt / dz;
    const int last_i = iz1 - 1;
    const int last_j = jx1 - 1;
    const int tile_rows = get_tile_rows();

    #pragma omp parallel
    {
        for (int ib = iz0; ib < last_i; ib += tile_rows) {
            int ie = ib + tile_rows;

            if (ie > last_i) {
                ie = last_i;
            }

            #pragma omp for schedule(static)
            for (int i = ib; i < ie; ++i) {
                stress_row(
                    vx,
                    vz,
                    sxx,
                    szz,
                    sxz,
                    lam,
                    lam2mu,
                    mu,
                    damp,
                    dtx,
                    dtz,
                    nx,
                    i,
                    jx0,
                    last_j
                );
            }

            int velocity_begin = ib;

            if (velocity_begin <= iz0) {
                velocity_begin = iz0 + 1;
            }

            #pragma omp for schedule(static)
            for (int i = velocity_begin; i < ie; ++i) {
                velocity_row(
                    vx,
                    vz,
                    sxx,
                    szz,
                    sxz,
                    inv_rho,
                    damp,
                    dtx,
                    dtz,
                    nx,
                    i,
                    jx0 + 1,
                    jx1
                );
            }
        }

        if (last_i > iz0) {
            #pragma omp for simd schedule(static)
            for (int j = jx0 + 1; j < jx1; ++j) {
                const int k = last_i * nx + j;
                const float dsxx_dx = sxx[k] - sxx[k - 1];
                const float dsxz_dz = sxz[k] - sxz[k - nx];
                const float dsxz_dx = sxz[k] - sxz[k - 1];
                const float dszz_dz = szz[k] - szz[k - nx];
                const float ir = inv_rho[k];
                const float vx_new = vx[k] + ir * (dtx * dsxx_dx + dtz * dsxz_dz);
                const float vz_new = vz[k] + ir * (dtx * dsxz_dx + dtz * dszz_dz);
                const float d = damp[k];
                vx[k] = vx_new * d;
                vz[k] = vz_new * d;
            }
        }
    }
}

void update_velocity_boundary(
    float * restrict vx,
    float * restrict vz,
    const float * restrict sxx,
    const float * restrict szz,
    const float * restrict sxz,
    const float * restrict inv_rho,
    const float * restrict damp,
    float dt,
    float dx,
    float dz,
    int nz,
    int nx,
    int iz0,
    int iz1,
    int jx0,
    int jx1
) {
    if (iz0 < 1) {
        iz0 = 1;
    }

    if (iz1 > nz - 1) {
        iz1 = nz - 1;
    }

    if (jx0 < 1) {
        jx0 = 1;
    }

    if (jx1 > nx - 1) {
        jx1 = nx - 1;
    }

    if (iz0 >= iz1 || jx0 >= jx1) {
        return;
    }

    const float dtx = dt / dx;
    const float dtz = dt / dz;

    #pragma omp parallel
    {
        #pragma omp for simd schedule(static)
        for (int j = jx0; j < jx1; ++j) {
            const int k = iz0 * nx + j;
            const float dsxx_dx = sxx[k] - sxx[k - 1];
            const float dsxz_dz = sxz[k] - sxz[k - nx];
            const float dsxz_dx = sxz[k] - sxz[k - 1];
            const float dszz_dz = szz[k] - szz[k - nx];
            const float ir = inv_rho[k];
            const float vx_new = vx[k] + ir * (dtx * dsxx_dx + dtz * dsxz_dz);
            const float vz_new = vz[k] + ir * (dtx * dsxz_dx + dtz * dszz_dz);
            const float d = damp[k];
            vx[k] = vx_new * d;
            vz[k] = vz_new * d;
        }

        #pragma omp for schedule(static)
        for (int i = iz0 + 1; i < iz1; ++i) {
            velocity_point(
                vx,
                vz,
                sxx,
                szz,
                sxz,
                inv_rho,
                damp,
                dtx,
                dtz,
                nx,
                i,
                jx0
            );
        }
    }
}