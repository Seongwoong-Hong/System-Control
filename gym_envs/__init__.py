from gym.envs.registration import register

register(
    id="CartPoleCont-v0",
    entry_point='gym_envs.envs:CartPoleContEnv'
)

register(
    id="CartPoleContTest-v0",
    entry_point='gym_envs.envs:CartPoleContTestEnv'
)

register(
    id='IP_custom-v0',
    entry_point='gym_envs.envs:IP_custom',
    )

register(
    id='IP_custom-v1',
    entry_point='gym_envs.envs:IP_custom_cont'
)

register(
    id='IP_custom-v2',
    entry_point='gym_envs.envs:IP_custom_PD'
)

register(
    id='IDP_custom-v0',
    entry_point='gym_envs.envs:IDP_custom',
    )
