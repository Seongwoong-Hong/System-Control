import subprocess

from common.path_config import MAIN_DIR

save_dir = MAIN_DIR / "RL" / "analysis" / "figure"
target_dir = MAIN_DIR / "RL" / "scripts" / "rlgames"


if __name__ == "__main__":
    script_path = str(MAIN_DIR / "RL" / "scripts" / "py" / "IsaacgymLearning.py")
    trial_type = "CoPSweep/noCurriculum90ATMax"
    checkpoint_base = target_dir / "runs" / trial_type
    for edptb in [0.08, 0.1, 0.12, 0.14, 0.16]:
        for upt in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            checkpoint_format = checkpoint_base / f"edptb{edptb:.2f}_upright_type{upt:.1f}" / "limLevel50_upright0" / "atm90_as250"
            file_paths = list(checkpoint_format.glob("IDP_*/nn/IDP.pth"))
            for idx, file_path in enumerate(file_paths):
                fig_path = save_dir / trial_type / f"edptb{edptb:.1f}_upright_type{upt:.1f}_{idx}.png"
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["python", script_path, f"checkpoint={str(file_path)}", f"fig_path={str(fig_path)}", "save_fig=True", "show_fig=False"])