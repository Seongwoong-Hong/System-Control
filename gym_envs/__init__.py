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

register(
    id="2DWorld-v0",
    entry_point='gym_envs.envs:TwoDWorldDetOrder',
    max_episode_steps=1000,
)

register(
    id='2DWorld-v1',
    entry_point='gym_envs.envs:TwoDWorldDet',
    max_episode_steps=100,
)

register(
    id='2DWorld-v2',
    entry_point='gym_envs.envs:TwoDWorld',
    max_episode_steps=100,
)

register(
    id='2DTarget-v2',
    entry_point='gym_envs.envs:TwoDTarget',
    max_episode_steps=100,
)

register(
    id='2DTarget-v0',
    entry_point='gym_envs.envs:TwoDTargetDet',
    max_episode_steps=100,
)


register(
    id='2DTarget_cont-v0',
    entry_point='gym_envs.envs:TwoDTargetCont',
    max_episode_steps=200,
)

register(
    id='2DTarget_cont-v1',
    entry_point='gym_envs.envs:TwoDTargetCont',
    max_episode_steps=100,
)

# environments based on mujoco
register(
    id='IP_MimicHuman-v0',
    entry_point='gym_envs.envs.mujoco:IPMimicHumanDet',
    max_episode_steps=360,
)

register(
    id='IP_MimicHuman-v2',
    entry_point='gym_envs.envs.mujoco:IPMimicHuman',
    max_episode_steps=90,
)

register(
    id='IP_MinEffort-v0',
    entry_point='gym_envs.envs.mujoco:IPMinEffortDet',
    max_episode_steps=360,
)

register(
    id='IP_MinEffort-v2',
    entry_point='gym_envs.envs.mujoco:IPMinEffort',
    max_episode_steps=90,
)

register(
    id='IDP_MimicHuman-v0',
    entry_point='gym_envs.envs.mujoco:IDPMimicHumanDet',
    max_episode_steps=360,
)


register(
    id='IDP_MimicHuman-v2',
    entry_point='gym_envs.envs.mujoco:IDPMimicHuman',
    max_episode_steps=90,
)

# environments based on pybullet

register(
    id='IDP_pybullet-v0',
    entry_point='gym_envs.envs.pybullet:InvertedDoublePendulumExpBulletEnv',
    max_episode_steps=600,
)

register(
    id='IDP_pybullet-v2',
    entry_point='gym_envs.envs.pybullet:InvertedDoublePendulumBulletEnv',
    max_episode_steps=600,
)

register(
    id='HPC_pybullet-v0',
    entry_point='gym_envs.envs.pybullet:HumanBalanceExpBulletEnv',
    max_episode_steps=599,
)

register(
    id='HPC_pybullet-v1',
    entry_point='gym_envs.envs.pybullet:HumanBalanceBulletEnv',
    max_episode_steps=599,
)

register(
    id='HPC_pbcrop-v0',
    entry_point='gym_envs.envs.pybullet:HumanBalanceExpBulletEnv',
    max_episode_steps=300,
)

register(
    id='HPC_pbcrop-v1',
    entry_point='gym_envs.envs.pybullet:HumanBalanceBulletEnv',
    max_episode_steps=300,
)
