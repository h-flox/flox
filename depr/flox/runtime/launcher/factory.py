from flox.runtime.launcher import (
    GlobusComputeLauncher,
    Launcher,
    LocalLauncher,
    ParslLauncher,
)


def create_launcher(kind: str, **launcher_cfg) -> Launcher:
    match kind:
        case "thread":
            return LocalLauncher(
                pool="thread", n_workers=launcher_cfg.get("max_workers", 3)
            )
        case "federation":
            return LocalLauncher(
                pool="federation", n_workers=launcher_cfg.get("max_workers", 3)
            )
        case "globus-compute":
            return GlobusComputeLauncher()
        case "parsl":
            return ParslLauncher(launcher_cfg)
        case _:
            raise ValueError("Illegal value for argument `kind`.")
