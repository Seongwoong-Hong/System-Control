clear, clc, close all
%===== REAL Environment is Cartpole =====%
% state: x, xdot, theta, thetadot (4)
% action: at cart, continuous (1)
env = rlPredefinedEnv("CartPole-Continuous");
init = env.reset;

T = 200;
x = zeros(4,T); u = zeros(1,T);
x(:,1) = init;
xhat = x; uhat = u;
K = zeros(T,4); k = zeros(T,1);

Tot = 500;
for iter = 1:Tot
    %===== RUN LQR =====%
    Vt = diag([1,1,1,1]); vt = diag([1,1,1,1])*xhat(:,end);
    for i = T:-1:1
        Ft = func_derv(xhat(:,i),uhat(:,i),env);
        Qt = diag([1,1,1,1,0.01]) + Ft'*Vt*Ft;
        qt = diag([1,1,1,1,0.01]) * [xhat(:,i);uhat(i)] + Ft'*vt;
        K(i,1:4) = -Qt(5,5)^-1*Qt(5,1:4);
        k(i) = -Qt(5,5)^-1*qt(5);
        Vt = Qt(1:4,1:4) + Qt(1:4,5)*K(i) + K(i)'*Qt(5,1:4) + K(i)'*Qt(5,5)*K(i);
        vt = qt(1:4) + Qt(1:4,5)*k(i) + K(i)'* qt(5) + K(i)'*Qt(5,5)*k(i);
    end
    %===== Forwarding Real Dynamics =====%
    x(:,1) = xhat(:,1);
    env.State = x(:,1);
    for i = 1:T-1
        u(i) = K(i,:)*(x(:,i) - xhat(:,i)) + k(i) + uhat(i);
        x(:,i+1) = env.step(u(i));
    end
    u(T) = 0;
    uhat = u; xhat = x;
    env.State = init;
end

env.State = xhat(:,1);
for i = 1:T
    x(1:4,i+1) = env.step(u(i));
    env.plot
%     if gifSave
%         frame = getframe(gcf);
%         img = frame2im(frame);
%         if i == 2
%             [imind, cm] = rgb2ind(img, 128);
%             imwrite(imind,cm,'bipedalWalking.gif','gif','Loopcount',Inf,'DelayTime', 0.1);
%         else
%             imind = rgb2ind(img, cm);
%             imwrite(imind,cm,'bipedalWalking.gif','gif','WriteMode','append','DelayTime',0.1);
%         end
%     end
end


function Fk = func_derv(xhat,uhat,env)
    %%%%%% TODO %%%%%%
    % Get partial difference of dynamics at xhat and uhat
    % input: xhat, uhat estimation of states and actions sequences from 0 to T
    % output: Ft, ct, Ct, partial diff. w.r.t. state and action in xhat, uhat
    del = 0.01 * env.Ts; q = [xhat; uhat];
    qn = model(q,env);
    qn1 = model(q + del*[1;0;0;0;0],env);
    qn2 = model(q + del*[0;1;0;0;0],env);
    qn3 = model(q + del*[0;0;1;0;0],env);
    qn4 = model(q + del*[0;0;0;1;0],env);
    qn5 = model(q + del*[0;0;0;0;1],env);
    Ak = [(qn1 - qn)/del, (qn2 - qn)/del, (qn3 - qn)/del, (qn4 - qn)/del];
    Bk = (qn5 - qn)/del;  
    Fk = [Ak, Bk];
end

function next_q = model(q,env)
    mc = env.MassCart; mp = env.MassPole; l = env.Length; g = env.Gravity; h = 0.1 * env.Ts;
    x = q(1); dx = q(2); th = q(3); dth = q(4); u = q(5);
    M = [mp + mc, mp*l*cos(th); mp*l*cos(th), mp*l*l];
    C = [0, -mp*l*dth*sin(th); 0, 0];
    G = [0; mp*l*g*sin(th)];
    B = [1; 0];
    
    next_d = [dx;dth] + h * M^-1 * (B*u - C*[dx;dth] - G);
    next_ = [x;th] + h * [dx;dth];
    next_q = [next_(1);next_d(1);next_(2);next_d(2)];
end