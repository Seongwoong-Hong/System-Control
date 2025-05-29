from matplotlib import pyplot as plt
from scipy import io
from tensorboard.backend.event_processing import event_accumulator

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
            trial_values = []
            for idx, file_path in enumerate(file_paths):
                mat_path = matlab_dir / trial_type / "upright" / f"{sweep_type}_{idx}" / "sub10"
                if not mat_path.is_dir():
                    mat_path.mkdir(exist_ok=True, parents=True)
                ea = event_accumulator.EventAccumulator(str(file_path))
                ea.Reload()
                events = ea.Scalars("episode_lengths/step")
                values = []
                steps = []
                for e in events:
                    values.append(e.value)
                    steps.append(e.step)

                data = {
                    'episode_length': values,
                    'steps': steps,
                }
                io.savemat(f"{mat_path}/tensorboard.mat", data)
