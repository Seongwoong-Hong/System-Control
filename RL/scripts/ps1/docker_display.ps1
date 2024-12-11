$WORKDIR = "/home/hsw/workspace/System-Control"
$WORKDIRWIN = "D:\hsw\HSW\inLabData\Workspace\System-Control"

$dockerArgs = @(
    "--name", "display",
    "--rm",
    "-it"
    "-u", "hsw:users",
    "-v", "${WORKDIRWIN}:${WORKDIR}",
    "-v", "\\wsl.localhost\Ubuntu\tmp\.X11-unix:/tmp/.X11-unix",
    "-w", "$WORKDIR",
    "--network", "host",
    "--env-file", "$WORKDIRWIN\RL\scripts\sh\docker_env_var.env",
    "-e", "DISPLAY=:0"
    "-e", "LD_PRELOAD=:/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so",
    "seongwoonghong/thesis:hpc-windows",
    "bash"
)

docker run @dockerArgs