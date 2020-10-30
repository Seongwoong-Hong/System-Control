close all

T  = 5;
h = 0.005;

mc = 5;
mp = 1;
l = 0.5;
g = 9.81;
x0 = [0; 0; -pi*1/4; 0];
u0 = 0;

Q = diag([1, 1, 1, 1]);
R = 0.001;

nx = numel(x0);
nu = numel(u0);

H = [mc+mp, mp*l; -1 -l];
C = -H\[0, 0; 0, g];
D = H\[1, 0]';
A = [1, h, 0, 0;
    C(1,1)*h, 1, C(1,2)*h, 0;
    0, 0, 1, h;
    C(2,1)*h, 0, C(2,2)*h, 1];
B = [0, D(1,1)*h, 0, D(2,1)*h]';

setup = struct('Q', Q, 'R', R, 'T', T, 'h', h, 'A', A, 'B', B, ...
    'mc', mc, 'mp', mp, 'l', l, 'g', g, 'nx', numel(x0), 'nu', numel(u0), ...
    'J0_max', 1e16);

Ks = BackwardLinLQR(setup);
[xs, us, J0] = ForwardLinLQR(x0, Ks, setup);

stackFig(1, xs(3,:), 'theta')
stackFig(2, us, 'controls')

t = linspace(0, T, T/h+1);
showCartPole(t, xs', 0)

function dxdt = SolCart(~,x,u,setup)
%     p = x(1);
    pdot = x(2);
    th = x(3);
    thdot = x(4);
    
    mc = setup.mc;
    mp = setup.mp;
    l = setup.l;
    g = setup.g;
    
    % H*qddot + C*qdot + E = D*u
    H = [mc+mp, mp*l*cos(th); -cos(th), -l];
    C = [0, -mp*l*thdot*sin(th); 0, 0];
    E = [0; g*sin(th)];
    D = [1; 0];
    
    dotdot = -H\C*[pdot; thdot] - H\E + H\D*u;
    
    dxdt = [pdot; dotdot(1); thdot; dotdot(2)];
end

function Ks = BackwardLinLQR(setup)
    
    Q = setup.Q;
    R = setup.R;
    T = setup.T;
    h = setup.h;
    nx = setup.nx;
    nu = setup.nu;
    A = setup.A;
    B = setup.B;
    
    F = [A, B];
    
    C = blkdiag(Q, R);
    V = zeros(size(Q));    
    Quus = zeros(nu, nu, T/h+1);
    Ks = zeros(nu, nx, T/h+1);
    
    for i = T/h+1:-1:1
        Qf = C + F'*V*F;
        Qxx = Qf(1:nx,1:nx);
        Qux = Qf(nx+nu,1:nx);
        Qxu = Qux';
        Quu = Qf(nx+nu,nx+nu);
        K = -Quu\Qux;
        V = Qxx + Qxu*K + K'*Qux + K'*Quu*K;
        Ks(:,:,i) = K;
        
        Quus(:,:,i) = Quu;
    end
end 

function [xs, us, J0] = ForwardLinLQR(x0, Ks, setup)
    
    T = setup.T;
    h = setup.h;
    Q = setup.Q;
    R = setup.R;
    J0_max = setup.J0_max;
    
    nx = setup.nx;
    nu = setup.nu;
    
    xs = zeros(nx, T/h+1);
    us = zeros(nu, T/h+1);
    xs(:,1) = x0;
    
    J0 = 0;
    
    options = odeset('RelTol',1e-6,'AbsTol',1e-9);
    for i = 1:T/h
        u = Ks(:,:,i)*xs(:,i);
        [~, x] = ode45(@(t,x) SolCart(t,x,u,setup), [0, h], xs(:,i), options);
        xs(:,i+1) = x(end,:);
        us(:,i) = u;
        
        J0 = J0 + 1/2*xs(:,1)'*Q*xs(:,1) + 1/2*u'*R*u;
        assert(J0 < J0_max, 'J blows up')
    end
end

function stackFig(fignum, data, str_title)
    h = figure(fignum);
    hold on, plot(data, 'k')
    if isempty(h.Children.Title.String)
        title(str_title)
    end
    lines = h.Children.Children;
    if numel(lines) > 10
        delete(lines(1))
    end
end