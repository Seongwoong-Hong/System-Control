import copy, torch
import numpy as np
from typing import Sequence
from imitation.data import rollout, types

def get_trajectories_probs(
        trajectories,
        policy,
        rng: np.random.RandomState = np.random
) -> Sequence[types.TrajectoryWithRew]:
    transitions = []
    for traj in trajectories:
        trans = copy.deepcopy(rollout.flatten_trajectories_with_rew([traj]))
        for i in range(trans.__len__()):
            obs = trans[i]['obs'].reshape(1, -1)
            acts = trans[i]['acts'].reshape(1, -1)
            latent_pi, _, latent_sde = policy._get_latent(torch.from_numpy(obs).to(policy.device))
            distribution = policy._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
            log_probs = distribution.log_prob(torch.from_numpy(acts).to(policy.device))
            trans[i]['infos']['log_probs'] = log_probs
        transitions.append(trans)
    return transitions