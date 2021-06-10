from gym.envs.registration import register, make

register(
    id="CartPoleCont-v0",
    entry_point='gym_envs.envs:CartPoleContEnv'
)

register(
    id="CartPoleContTest-v0",
    entry_point='gym_envs.envs:CartPoleContTestEnv'
)

register(
    id='IDP_classic-v0',
    entry_point='gym_envs.envs:InvertedDoublePendulum',
    max_episode_steps=600,
)

# environments based on mujoco
# expert: v0, agent: v1
# v2 environment is not used yet
register(
    id='IP_custom-v0',
    entry_point='gym_envs.envs.mujoco:IPCustomExp',
    max_episode_steps=600,
)

register(
    id='IP_custom-v1',
    entry_point='gym_envs.envs.mujoco:IPCustom',
    max_episode_steps=600,
)

register(
    id='IP_custom-v2',
    entry_point='gym_envs.envs.mujoco:IPCustom',
    max_episode_steps=600,
)


register(
    id='IDP_custom-v0',
    entry_point='gym_envs.envs.mujoco:IDPCustomExp',
    max_episode_steps=600,
)

register(
    id='IDP_custom-v1',
    entry_point='gym_envs.envs.mujoco:IDPCustom',
    max_episode_steps=600,
)

register(
    id='IDP_custom-v2',
    entry_point='gym_envs.envs.mujoco:IDPCustomEasy',
    max_episode_steps=600,
)

register(
    id='HPC_custom-v0',
    entry_point='gym_envs.envs.mujoco:IDPHumanExp',
    max_episode_steps=599,
)

register(
    id='HPC_custom-v1',
    entry_point='gym_envs.envs.mujoco:IDPHuman',
    max_episode_steps=599,
)

