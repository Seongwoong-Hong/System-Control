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
    id='DiscretizedPendulum-v2',
    entry_point='gym_envs.envs:DiscretizedPendulum',
    max_episode_steps=50,
)

register(
    id='DiscretizedPendulum-v0',
    entry_point='gym_envs.envs:DiscretizedPendulumDet',
    max_episode_steps=50,
)

register(
    id='DiscretizedDoublePendulum-v2',
    entry_point='gym_envs.envs:DiscretizedDoublePendulum',
    max_episode_steps=50,
)

register(
    id='DiscretizedDoublePendulum-v0',
    entry_point='gym_envs.envs:DiscretizedDoublePendulumDet',
    max_episode_steps=50,
)

register(
    id='DiscretizedHuman-v2',
    entry_point='gym_envs.envs:DiscretizedHuman',
    max_episode_steps=50,
)

register(
    id='DiscretizedHuman-v0',
    entry_point='gym_envs.envs:DiscretizedHumanDet',
    max_episode_steps=50,
)

register(
    id="2DWorld-v0",
    entry_point='gym_envs.envs:TwoDWorldDetOrder',
    max_episode_steps=100,
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
    id='2DWorld_disc-v0',
    entry_point='gym_envs.envs:TwoDWorldDiscDet',
    max_episode_steps=20,
)

register(
    id='2DWorld_disc-v2',
    entry_point='gym_envs.envs:TwoDWorldDisc',
    max_episode_steps=20,
)

register(
    id='SpringBall_disc-v2',
    entry_point='gym_envs.envs:SpringBallDisc',
    max_episode_steps=40,
)

register(
    id='SpringBall_disc-v0',
    entry_point='gym_envs.envs:SpringBallDiscDet',
    max_episode_steps=40,
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
    id='2DTarget_disc-v0',
    entry_point='gym_envs.envs:TwoDTargetDiscDet',
    max_episode_steps=200,
)

register(
    id='2DTarget_disc-v2',
    entry_point='gym_envs.envs:TwoDTargetDisc',
    max_episode_steps=200,
)

register(
    id='1DTarget_disc-v2',
    entry_point='gym_envs.envs:OneDTargetDisc',
    max_episode_steps=200,
)

register(
    id='1DTarget_disc-v0',
    entry_point='gym_envs.envs:OneDTargetDiscDet',
    max_episode_steps=200,
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
# expert: v0, agent: v1
# v2 environment is not used yet
register(
    id='IP_custom-v0',
    entry_point='gym_envs.envs.mujoco:IPCustomExp',
    max_episode_steps=50,
)

register(
    id='IP_custom-v1',
    entry_point='gym_envs.envs.mujoco:IPCustom',
    max_episode_steps=50,
)

register(
    id='IP_custom-v2',
    entry_point='gym_envs.envs.mujoco:IPCustom',
    max_episode_steps=50,
)


register(
    id='IDP_custom-v0',
    entry_point='gym_envs.envs.mujoco:IDPCustomExp',
    max_episode_steps=50,
)

register(
    id='IDP_custom-v2',
    entry_point='gym_envs.envs.mujoco:IDPCustom',
    max_episode_steps=50,
)

register(
    id='HPC_custom-v0',
    entry_point='gym_envs.envs.mujoco:IDPHumanExp',
    max_episode_steps=50,
)

register(
    id='HPC_custom-v2',
    entry_point='gym_envs.envs.mujoco:IDPHuman',
    max_episode_steps=50,
)

register(
    id="HPC_crop-v0",
    entry_point='gym_envs.envs.mujoco:IDPHumanExp',
    max_episode_steps=300,
)

register(
    id='HPC_crop-v1',
    entry_point='gym_envs.envs.mujoco:IDPHuman',
    max_episode_steps=300,
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
