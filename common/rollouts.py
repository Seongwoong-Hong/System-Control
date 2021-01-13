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
            obs = torch.from_numpy(trans[i]['obs'].reshape(1, -1)).to(policy.device)
            acts = torch.from_numpy(trans[i]['acts'].reshape(1, -1)).to(policy.device)
            latent_pi, _, latent_sde = policy._get_latent(obs)
            distribution = policy._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
            log_probs = distribution.log_prob(acts)
            trans[i]['infos']['log_probs'], trans[i]['infos']['rwinp'] = log_probs, torch.cat((obs, acts), dim=1)
        transitions.append(trans)
    return transitions