import os
from contextlib import nullcontext

def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")

FWMP_SCOREP = env_bool("FWMP_SCOREP", default=False)
scorep_user = None
scorep_available = False
if FWMP_SCOREP:
    scorep_root = os.environ.get("SCOREP_EXPERIMENT_DIRECTORY", "..")
    os.makedirs(scorep_root, exist_ok=True)
    os.environ["SCOREP_EXPERIMENT_DIRECTORY"] = scorep_root
    try:
        import scorep.user as scorep_user
        scorep_available = True
    except Exception:
        scorep_user = None
        scorep_available = False

def scorep_region(name):
    if FWMP_SCOREP and scorep_available and hasattr(scorep_user, "region"):
        return scorep_user.region(name)
    return nullcontext()
