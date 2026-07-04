import os
import ctypes
import numpy as np

_float2 = np.ctypeslib.ndpointer(
    dtype=np.float32,
    ndim=2,
    flags="C_CONTIGUOUS",
)

class ElasticKernels:
    def __init__(self, library_path=None):
        if library_path is None:
            library_path = os.environ.get("FWMP_KERNEL_LIB")
        if library_path is None:
            package_dir = os.path.dirname(__file__)
            root_dir = os.path.dirname(package_dir)
            library_path = os.path.join(root_dir, "libelastic_kernels.so")
        self.library_path = library_path
        self.lib = ctypes.CDLL(library_path)
        self._setup_argtypes()

    def _setup_argtypes(self):
        self.lib.update_stress.argtypes = [
            _float2, _float2,
            _float2, _float2, _float2,
            _float2, _float2, _float2, _float2,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        self.lib.update_stress.restype = None
        self.lib.update_velocity.argtypes = [
            _float2, _float2,
            _float2, _float2, _float2,
            _float2, _float2,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        self.lib.update_velocity.restype = None

    def update_stress(
        self,
        vx,
        vz,
        sxx,
        szz,
        sxz,
        lam,
        lam2mu,
        mu,
        damp,
        dt,
        dx,
        dz,
        iz0,
        iz1,
        jx0,
        jx1,
    ):
        self.lib.update_stress(
            vx, vz,
            sxx, szz, sxz,
            lam, lam2mu, mu, damp,
            np.float32(dt), np.float32(dx), np.float32(dz),
            vx.shape[0], vx.shape[1],
            iz0, iz1,
            jx0, jx1,
        )

    def update_velocity(
        self,
        vx,
        vz,
        sxx,
        szz,
        sxz,
        inv_rho,
        damp,
        dt,
        dx,
        dz,
        iz0,
        iz1,
        jx0,
        jx1,
    ):
        self.lib.update_velocity(
            vx, vz,
            sxx, szz, sxz,
            inv_rho, damp,
            np.float32(dt), np.float32(dx), np.float32(dz),
            vx.shape[0], vx.shape[1],
            iz0, iz1,
            jx0, jx1,
        )
