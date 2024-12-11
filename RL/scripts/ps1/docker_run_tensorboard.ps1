$WORKDIR = "/home/hsw/workspace/System-Control"
$WORKDIRWIN = "D:\hsw\HSW\inLabData\Workspace\System-Control"

$dockerArgs = @(
    "--name", "tensorboard",
    "--rm",
    "-it"
    "-v", "${WORKDIRWIN}:${WORKDIR}",
    "-w", "$WORKDIR",
    "-u", "hsw:users",
    "-p", "6006:6006",
    "-p", "6007:6007",
    "-p", "6008:6008",
    "--env-file", "$WORKDIRWIN\RL\scripts\sh\docker_env_var.env",
    "seongwoonghong/thesis:hpc-latest",
    "bash"
)

docker run @dockerArgs