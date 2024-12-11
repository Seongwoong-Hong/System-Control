param(
    [string]$HEADIP = "head"
)

$WORKDIR = "/home/hsw/workspace/System-Control"
$WORKDIRWIN = "D:\hsw\HSW\inLabData\Workspace\System-Control"

if ($HEADIP -eq "head" -or $HEADIP -eq "HEAD" -or $HEADIP -eq "Head") {
    $CMDL = "ray start --head --num-gpus=1"
} else {
    $CMDL = "ray start --num-gpus=1 --address=$HEADIP:6379"
}

docker run `
  --name "ray_worker" `
  --rm `
  -it `
  --gpus all `
  -v "${WORKDIRWIN}:${WORKDIR}" `
  -w $WORKDIR `
  -u "hsw:users" `
  --env-file "$WORKDIRWIN\RL\scripts\sh\docker_env_var.env" `
  --network host `
  --shm-size="16.00gb" `
  seongwoonghong/thesis:hpc-latest `
  bash -c "$CMDL; exec /bin/bash"