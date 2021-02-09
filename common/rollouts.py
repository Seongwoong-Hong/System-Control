import copy
import torch
from typing import Sequence

from imitation.data import rollout


def get_trajectories_probs(
        trajectories,
        policy,
) -> Sequence[torch.Tensor]:
    transitions = []
    for traj in trajectories:
        trans = copy.deepcopy(rollout.flatten_trajectories_with_rew([traj]))
        obs = torch.from_numpy(trans[0]['obs'].reshape(1, -1)).to(policy.device)
        acts = torch.from_numpy(trans[0]['acts'].reshape(1, -1)).to(policy.device)
        log_probs = policy.get_log_prob_from_act(obs, acts)
        concats = torch.cat((obs, acts, log_probs.reshape(-1, 1)), dim=1)
        for i in range(len(trans)-1):
            obs = torch.from_numpy(trans[i+1]['obs'].reshape(1, -1)).to(policy.device)
            acts = torch.from_numpy(trans[i+1]['acts'].reshape(1, -1)).to(policy.device)
            log_probs = policy.get_log_prob_from_act(obs, acts)
            concats = torch.cat((concats, torch.cat((obs, acts, log_probs.reshape(-1, 1)), dim=1)), dim=0)
        transitions += [concats]
    return transitions
