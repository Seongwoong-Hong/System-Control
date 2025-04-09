import subprocess

from common.path_config import MAIN_DIR

save_dir = MAIN_DIR / "RL" / "analysis" / "figure"
target_dir = MAIN_DIR / "RL" / "scripts" / "rlgames"


if __name__ == "__main__":
    script_path = str(MAIN_DIR / "RL" / "scripts" / "py" / "IsaacgymLearning.py")
    trial_type = "TorqueSweep/trrSweep"
    checkpoint_base = target_dir / "runs" / trial_type
    for sw1 in [0.01, 0.03, 0.05, 0.07, 0.09]:
        for sw2 in [0.01, 0.02, 0.03, 0.04, 0.05]:
            sweep_type = f"avg_coeff{sw1:.2f}_tqrate_ratio{sw2:.2f}"
            checkpoint_format = checkpoint_base / sweep_type / "limLevel50_upright0" / "atm90_as250"
            file_paths = list(checkpoint_format.glob("IDP_*/nn/IDP.pth"))
            for idx, file_path in enumerate(file_paths):
                fig_path = save_dir / trial_type / "lean" / f"{sweep_type}_{idx}.png"
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["python", script_path, f"checkpoint={str(file_path)}", f"fig_path={str(fig_path)}", "save_fig=True", "show_fig=False", f"avg_coeff={sw1}"])