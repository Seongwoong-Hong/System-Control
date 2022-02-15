clear, clc, close all
%===== REAL Environment is Cartpole =====%
% state: x, xdot, theta, thetadot (4)
% action: at cart, continuous (1)
env = rlPredefinedEnv("CartPole-Continuous");

T = 200;
x = zeros(4,T); u = zeros(1,T);
x(:,1) = env.reset;
K = zeros(T,4); k = zeros(T,1);
iter = 0;

%===== RUN LQR =====%
Vt = diag([1,1,1,1]);
mc = env.MassCart; mp = env.MassPole; l = env.Length; g = env.Gravity; h = env.Ts;
Ft = [eye(4,4), zeros(4,1)] + h * [0 1 0 0 0; -mp*g/mc, 0, 0, 0, 1/mc; 0, 0, 0, 1, 0; 0, 0, -(mp+mc)*g/(mc*l), 0, -1/(mc*l)];
for i = T:-1:1
    Qt = diag([1,1,1,1,0.01]) + Ft'*Vt*Ft;
    K(i,1:4) = -Qt(5,5)^-1*Qt(5,1:4);
    Vt = Qt(1:4,1:4) + Qt(1:4,5)*K(i) + K(i)'*Qt(5,1:4) + K(i)'*Qt(5,5)*K(i);
end

%===== Forwarding Real Dynamics =====%
for i = 1:T
    u(i) = K(i,:)*x(:,i);
    x(:,i+1) = env.step(u(i));
    h = plot(env);
end