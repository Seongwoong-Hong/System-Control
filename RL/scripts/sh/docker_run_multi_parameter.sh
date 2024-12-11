WORKDIR=/home/hsw/workspace

docker run \
--name cost_ratio_tuning2 \
--rm \
-v ~/Workspace/System-Control:$WORKDIR \
-w $WORKDIR \
-u hsw:hsw \
--env-file docker_env_var.env \
seongwoonghong/thesis:hpc-latest \
sh $WORKDIR/RL/scripts/learn_PolicyLearning.sh