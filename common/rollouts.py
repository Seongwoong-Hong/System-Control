import copy
import torch
from typing import Sequence, List

import numpy as np
from imitation.data import rollout, types
from imitation.data.types import TransitionsWithRew


def get_trajectories_probs(
        trajectories,
        policy,
        rng: np.random.RandomState = np.random
) -> Sequence[types.TransitionsWithRew]:
    transitions = []
    for traj in trajectories:
        trans = copy.deepcopy(rollout.flatten_trajectories_with_rew([traj]))
        for i in range(trans.__len__()):
            obs = torch.from_numpy(trans[i]['obs'].reshape(1, -1)).to(policy.device)
            acts = torch.from_numpy(trans[i]['acts'].reshape(1, -1)).to(policy.device)
            log_probs = policy.get_log_prob_from_act(obs, acts)
            trans[i]['infos']['log_probs'], trans[i]['infos']['rwinp'] = log_probs, torch.cat((obs, acts), dim=1).reshape(-1)
        transitions.append(trans)
    return transitions
