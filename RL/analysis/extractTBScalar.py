from tensorboard.backend.event_processing import event_accumulator

log_dir = "/home/hsw/workspace/System-Control/RL/scripts/rlgames/runs/TorqueSweepSlow/limLevel50_upright0/atm90_as250/IDP_2025-04-20_08-16-17/summaries/events.out.tfevents.1745136977.lab7040"

from common.path_config import MAIN_DIR

figure_dir = MAIN_DIR / "RL" / "analysis" / "figure"
matlab_dir = MAIN_DIR / "RL" / "analysis" / "MATLAB" / "IDPMinEffort"
target_dir = MAIN_DIR / "RL" / "scripts" / "rlgames"

if __name__ == "__main__":
    trial_type = "TorqueSweepSlow"
    checkpoint_base = target_dir / "runs" / trial_type
    for sw1 in [0.1]:
        for sw2 in [0.0]:
            sweep_type = f"avg_coeff{sw1:.2f}_tqrate_ratio{sw2:.2f}"
            checkpoint_format = checkpoint_base / sweep_type / "limLevel50_upright0" / "atm90_as250"
            file_paths = list(checkpoint_format.glob("IDP_*/summaries/events.out.tfevents.*"))
            for idx, file_path in enumerate(file_paths):
                ea = event_accumulator.EventAccumulator(str(file_path))
                ea.Reload()
                events = ea.Scalars("rewards/step")
                values = []
                steps = []
                for e in events:
                    values.append(e.value)
                    steps.append(e.step)
