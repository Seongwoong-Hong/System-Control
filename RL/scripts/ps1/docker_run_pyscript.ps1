param(
    [string]$BaseName = "IsaacgymLearning",
    [switch]$gpus
)

$WORKDIR = "/home/hsw/workspace/System-Control"
$WORKDIRWIN = "D:\hsw\HSW\inLabData\Workspace\System-Control"
$COUNTER = 1
$CONT_NAME = $BaseName + $COUNTER

while ((docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $CONT_NAME }) -ne $null) {
    $COUNTER++
    $CONT_NAME = $BaseName + $COUNTER
}
Write-Host "$CONT_NAME"
$dockerArgs = @(
    "--name", "$CONT_NAME",
    "-it",
    "--rm",
    "-v", "${WORKDIRWIN}:${WORKDIR}",
    "-w", "$WORKDIR",
    "-u", "hsw:users",
    "--env-file", "$WORKDIRWIN\RL\scripts\sh\docker_env_var.env",
    "--gpus", "all",
    "seongwoonghong/thesis:hpc-isaacgym",
    "python", "$WORKDIR/RL/scripts/py/$BaseName.py"
)

docker run @dockerArgs