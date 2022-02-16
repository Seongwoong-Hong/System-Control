clear, clc, close all
%===== REAL Environment is Cartpole =====%
% state: x, xdot, theta, thetadot (4)
% action: at cart, continuous (1)
env = rlPredefinedEnv("CartPole-Continuous");

mc = env.MassCart; mp = env.MassPole; l = env.Length; g = env.Gravity; h = env.Ts;
A = [0, 1, 0, 0; -mp*g/mc, 0, 0, 0; 0, 0, 0, 1; 0, 0, -(mp+mc)*g/(mc*l), 0];
B = [0; 1/mc; 0; -1/(mc*l)];
Q = diag([1,1,1,1]); R = 0.01; N = 0;
[K, S, P] = lqr(A,B,Q,R,N);
x(:,1) = env.reset;
for i = 1:200
    u(i) = -K*x(:,i);
    x(:,i+1) = env.step(u(i));
    env.plot
end