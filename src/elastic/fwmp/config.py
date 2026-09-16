import os
from dataclasses import dataclass

@dataclass
class Config:
    n_iterations: int
    frame_stride: int
    direct_vz_source: int
    ds: int
    output_batch: int
    nb: int
    slurm_job_id: str
    base_output_dir: str
    vp_path: str
    vs_path: str
    rho_path: str

def load_config():
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "noslurm")
    base_output_dir = os.environ.get(
        "FWMP_BASE_OUTPUT_DIR",
        os.path.join("..", "output", f"job_{slurm_job_id}"),
    )
    return Config(
        n_iterations=int(os.environ.get("FWMP_NITER", "50000")),
        frame_stride=int(os.environ.get("FWMP_FRAME_STRIDE", "100")),
        direct_vz_source=int(os.environ.get("FWMP_DIRECT_VZ_SOURCE", "0")),
        ds=int(os.environ.get("FWMP_DS", "10")),
        output_batch=int(os.environ.get("FWMP_OUTPUT_BATCH", "8")),
        nb=int(os.environ.get("FWMP_NB", "240")),
        slurm_job_id=slurm_job_id,
        base_output_dir=base_output_dir,
        vp_path=os.environ.get(
            "FWMP_VP_PATH",
            "../data/MODEL_P-WAVE_VELOCITY_1.25m.segy",
        ),
        vs_path=os.environ.get(
            "FWMP_VS_PATH",
            "../data/MODEL_S-WAVE_VELOCITY_1.25m.segy",
        ),
        rho_path=os.environ.get(
            "FWMP_RHO_PATH",
            "../data/MODEL_DENSITY_1.25m.segy",
        ),
    )
