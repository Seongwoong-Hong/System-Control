import hydra
import time
from omegaconf import DictConfig

@hydra.main(config_path="cfg", config_name="config.yaml")
def main(cfg: DictConfig):
    print(f"Running SLURM job with config: {cfg}")
    time.sleep(10)

if __name__ == "__main__":
    main()
