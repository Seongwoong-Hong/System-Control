**선행 수행 사항**
- isaacgym 과 IsaacGymEnvs 패키지가 필요합니다.
- 현재 폴더에 isaacgym/ 및 IsaacGymEnvs/ 폴더명으로 각 패키지가 존재해야 합니다.
- isaacgym: https://developer.nvidia.com/isaac-gym
- IsaacGymEnvs: https://github.com/isaac-sim/IsaacGymEnvs

**build 이후 sudo가 필요한 경우**
- docker build 후, ${USERNAME} 계정의 패스워드 설정이 완료되어야 sudo 사용이 가능합니다.
- docker run 혹은 exec 의 -u 옵션을 사용해 root 계정으로 접속후, passwd 커맨드를 사용하여 사용자의 비밀번호를 설정합니다.
- 이후 컨테이너를 image로 커밋하면 sudo 사용을 위한 설정이 완료됩니다.