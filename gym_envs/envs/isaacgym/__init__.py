from isaacgymenvs.tasks import isaacgym_task_map, Ant, Cartpole

from gym_envs.envs.isaacgym.double_inverted_pendulum import *

isaacgym_task_map["IDPMinEffort"] = IDPMinEffort
isaacgym_task_map["IDPMinEffortHumanLeanDet"] = IDPMinEffortHumanLeanDet
isaacgym_task_map["IDPMinEffortDet"] = IDPMinEffortDet
isaacgym_task_map["IDPMinEffortHumanDet"] = IDPMinEffortHumanDet
isaacgym_task_map["IDPLeanAndRelease"] = IDPLeanAndRelease
isaacgym_task_map["IDPLeanAndReleaseDet"] = IDPLeanAndReleaseDet
isaacgym_task_map["IDPForwardPushDet"] = IDPForwardPushDet
isaacgym_task_map["CartpoleDet"] = Cartpole
isaacgym_task_map["AntDet"] = Ant