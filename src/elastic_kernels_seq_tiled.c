#include <stdlib.h>

#ifndef ELASTIC_TILE_Z
#define ELASTIC_TILE_Z 8
#endif

#ifndef ELASTIC_TILE_X
#define ELASTIC_TILE_X 256
#endif

static int tile_z_value(void) {
    static int initialized = 0;
    static int value = ELASTIC_TILE_Z;

    if (!initialized) {
        const char *text = getenv("FWMP_TILE_Z");

        if (text != NULL) {
            const int requested = atoi(text);

            if (requested > 0) {
                value = requested;
            }
        }

        initialized = 1;
    }

    return value;
}

static int tile_x_value(void) {
    static int initialized = 0;
    static int value = ELASTIC_TILE_X;

    if (!initialized) {
        const char *text = getenv("FWMP_TILE_X");

        if (text != NULL) {
            const int requested = atoi(text);

            if (requested > 0) {
                value = requested;
            }
        }

        initialized = 1;
    }

    return value;
}

void update_stress_velocity_seq_tiled(
    float *restrict vx,
    float *restrict vz,
    float *restrict sxx,
    float *restrict szz,
    float *restrict sxz,
    const float *restrict lam,
    const float *restrict lam2mu,
    const float *restrict mu,
    const float *restrict inv_rho,
    const float *restrict damp,
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
    const int tile_z = tile_z_value();
    const int tile_x = tile_x_value();

    for (int ib = iz0; ib < iz1; ib += tile_z) {
        int ie = ib + tile_z;

        if (ie > iz1) {
            ie = iz1;
        }

        for (int jb = jx0; jb < jx1; jb += tile_x) {
            int je = jb + tile_x;

            if (je > jx1) {
                je = jx1;
            }

            for (int i = ib; i < ie; ++i) {
                const int row = i * nx;
                const int rowp = row + nx;

                for (int j = jb; j < je; ++j) {
                    const int k = row + j;

                    const float dvx_dx = vx[k + 1] - vx[k];
                    const float dvx_dz = vx[rowp + j] - vx[k];
                    const float dvz_dx = vz[k + 1] - vz[k];
                    const float dvz_dz = vz[rowp + j] - vz[k];

                    const float sxx_new =
                        sxx[k]
                        + lam2mu[k] * dtx * dvx_dx
                        + lam[k] * dtz * dvz_dz;

                    const float szz_new =
                        szz[k]
                        + lam[k] * dtx * dvx_dx
                        + lam2mu[k] * dtz * dvz_dz;

                    const float sxz_new =
                        sxz[k]
                        + mu[k] * (
                            dtz * dvx_dz
                            + dtx * dvz_dx
                        );

                    const float d = damp[k];

                    sxx[k] = sxx_new * d;
                    szz[k] = szz_new * d;
                    sxz[k] = sxz_new * d;
                }
            }

            for (int i = ib; i < ie; ++i) {
                const int row = i * nx;
                const int rowm = row - nx;

                for (int j = jb; j < je; ++j) {
                    const int k = row + j;

                    const float dsxx_dx =
                        sxx[k] - sxx[k - 1];

                    const float dsxz_dz =
                        sxz[k] - sxz[rowm + j];

                    const float dsxz_dx =
                        sxz[k] - sxz[k - 1];

                    const float dszz_dz =
                        szz[k] - szz[rowm + j];

                    const float ir = inv_rho[k];

                    const float vx_new =
                        vx[k]
                        + ir * (
                            dtx * dsxx_dx
                            + dtz * dsxz_dz
                        );

                    const float vz_new =
                        vz[k]
                        + ir * (
                            dtx * dsxz_dx
                            + dtz * dszz_dz
                        );

                    const float d = damp[k];

                    vx[k] = vx_new * d;
                    vz[k] = vz_new * d;
                }
            }
        }
    }
}
