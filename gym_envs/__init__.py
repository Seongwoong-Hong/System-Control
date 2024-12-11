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
    id='IDPPD_MimicHuman-v0',
    entry_point='gym_envs.envs.mujoco:IDPPDMimicHumanDet',
    max_episode_steps=360,
)


register(
    id='IDPPD_MimicHuman-v2',
    entry_point='gym_envs.envs.mujoco:IDPPDMimicHuman',
    max_episode_steps=90,
)

register(
    id='IDPPD_MinEffort-v0',
    entry_point='gym_envs.envs.mujoco:IDPPDMinEffortDet',
    max_episode_steps=360,
)


register(
    id='IDPPD_MinEffort-v2',
    entry_point='gym_envs.envs.mujoco:IDPPDMinEffort',
    max_episode_steps=90,
)

register(
    id='IDPPD_MinMetCost-v0',
    entry_point='gym_envs.envs.mujoco:IDPPDMinEffortDet',
    max_episode_steps=360,
)

register(
    id='IDPPD_MinMetCost-v2',
    entry_point='gym_envs.envs.mujoco:IDPPDMinMetCost',
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
    max_episode_steps=360,
)

register(
    id='IDP_MinEffort-v0',
    entry_point='gym_envs.envs.mujoco:IDPMinEffortDet',
    max_episode_steps=360,
)


register(
    id='IDP_MinEffort-v2',
    entry_point='gym_envs.envs.mujoco:IDPMinEffort',
    max_episode_steps=600,
)

register(
    id='IDP_MinMetCost-v0',
    entry_point='gym_envs.envs.mujoco:IDPMinEffortDet',
    max_episode_steps=360,
)

register(
    id='IDP_MinMetCost-v2',
    entry_point='gym_envs.envs.mujoco:IDPMinMetCost',
    max_episode_steps=600,
)

register(
    id='IDP_ForwardPush-v0',
    entry_point='gym_envs.envs.mujoco:IDPForwardPushDet',
    max_episode_steps=360,
)

register(
    id='IDP_HeadTrack-v2',
    entry_point='gym_envs.envs.mujoco:IDPHeadTrack',
    max_episode_steps=1200,
)

register(
    id='IDP_HeadTrack-v0',
    entry_point='gym_envs.envs.mujoco:IDPHeadTrackDet',
    max_episode_steps=360,
)

register(
    id='IDP_SinPtb-v2',
    entry_point='gym_envs.envs.mujoco:IDPSinPtb',
    max_episode_steps=360,
)

register(
    id='IDP_SinPtb-v0',
    entry_point='gym_envs.envs.mujoco:IDPSinPtbDet',
    max_episode_steps=360,
)

register(
    id='IDP_InitState-v2',
    entry_point='gym_envs.envs.mujoco:IDPInitState',
    max_episode_steps=360,
)

register(
    id='IDP_InitState-v0',
    entry_point='gym_envs.envs.mujoco:IDPInitState',
    max_episode_steps=360,
)

register(
    id='Cartpole-v2',
    entry_point='gym_envs.envs.mujoco:CartpoleEnv',
    max_episode_steps=500,
)

register(
    id='Cartpole-v0',
    entry_point='gym_envs.envs.mujoco:CartpoleEnv',
    max_episode_steps=500,
)

register(
    id='humanoid-v0',
    entry_point='gym_envs.envs.mujoco:HumanoidEnv',
    max_episode_steps=360,
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
