import subprocess

from common.path_config import MAIN_DIR

figure_dir = MAIN_DIR / "RL" / "analysis" / "figure"
matlab_dir = MAIN_DIR / "RL" / "analysis" / "MATLAB" / "IDPMinEffort"
target_dir = MAIN_DIR / "RL" / "scripts" / "rlgames"


if __name__ == "__main__":
    script_path = str(MAIN_DIR / "RL" / "scripts" / "py" / "IsaacgymLearning.py")
    trial_type = "CoPSweep/noCurriculum90ATMax"
    checkpoint_base = target_dir / "runs" / trial_type
    for sw1 in [0.08, 0.10]:
        for sw2 in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
            sweep_type = f"edptb{sw1:.2f}_upright_type{sw2:.1f}"
            checkpoint_format = checkpoint_base / sweep_type / "limLevel50_upright0" / "atm90_as250"
            file_paths = list(checkpoint_format.glob("IDP_*/nn/IDP.pth"))
            for idx, file_path in enumerate(file_paths):
                fig_path = figure_dir / trial_type / "upright" / f"{sweep_type}_{idx}.png"
                mat_path = matlab_dir / trial_type / "upright" / f"{sweep_type}_{idx}" / "sub10"

                fig_path.parent.mkdir(parents=True, exist_ok=True)
                mat_path.mkdir(parents=True, exist_ok=True)

                subprocess.run([
                    "python", script_path, f"checkpoint={str(file_path)}", "show_fig=False",
                     f"fig_path={str(fig_path)}", "save_fig=True",
                     f"mat_path={str(mat_path)}", "save_mat=True",
                ])
