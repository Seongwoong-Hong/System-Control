from gym.envs.registration import register, make

register(
    id="CartPoleCont-v0",
    entry_point='gym_envs.envs:CartPoleContEnv'
)

register(
    id="CartPoleContTest-v0",
    entry_point='gym_envs.envs:CartPoleContTestEnv'
)
# expert: v0, agent: v1
# v2 environment is not used yet
register(
    id='IP_custom-v0',
    entry_point='gym_envs.envs:IPCustomExp'
)

register(
    id='IP_custom-v1',
    entry_point='gym_envs.envs:IPCustom',
)

register(
    id='IP_custom-v2',
    entry_point='gym_envs.envs:IPCustom',
    max_episode_steps=600,
)


register(
    id='IDP_custom-v0',
    entry_point='gym_envs.envs:IDPCustomExp',
)

register(
    id='IDP_custom-v1',
    entry_point='gym_envs.envs:IDPCustom',
)

register(
    id='IDP_custom-v2',
    entry_point='gym_envs.envs:IDPCustomEasy',
    max_episode_steps=600,
)

register(
    id='HPC_custom-v0',
    entry_point='gym_envs.envs:IDPHumanExp',
)

register(
    id='HPC_custom-v1',
    entry_point='gym_envs.envs:IDPHuman',
)
