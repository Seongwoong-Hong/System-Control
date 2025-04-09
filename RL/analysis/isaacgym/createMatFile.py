import subprocess

from common.path_config import MAIN_DIR

save_dir = MAIN_DIR / "RL" / "analysis" / "MATLAB" / "IDPMinEffort"
target_dir = MAIN_DIR / "RL" / "scripts" / "rlgames"


if __name__ == "__main__":
    script_path = str(MAIN_DIR / "RL" / "scripts" / "py" / "IsaacgymLearning.py")
    trial_type = "CoPSweep/noCurriculum90ATMax"
    checkpoint_base = target_dir / "runs" / trial_type
    for sw1 in [0.08, 0.10, 0.12, 0.14, 0.16]:
        for sw2 in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            sweep_type = f"edptb{sw1:.2f}_upright_type{sw2:.1f}"
            checkpoint_format = checkpoint_base / sweep_type / "limLevel50_upright0" / "atm90_as250"
            file_paths = list(checkpoint_format.glob("IDP_*/nn/IDP.pth"))
            for idx, file_path in enumerate(file_paths):
                mat_path = save_dir / trial_type / "upright" / f"{sweep_type}_{idx}" / "sub10"
                mat_path.mkdir(parents=True, exist_ok=True)
                subprocess.run(["python", script_path, f"checkpoint={str(file_path)}", f"avg_coeff=0.05", f"mat_path={str(mat_path)}", "save_mat=True", "show_fig=False"])
