close all

T  = 0.5;
h = 0.005;

mc = 5;
mp = 1;
l = 0.5;
g = 9.81;
x0 = [0; 0; pi/12; 0];
u0 = 0;

Q = diag([0, 0, 1, 1]);
R = 0.001;

nx = numel(x0);
nu = numel(u0);

setup = struct('mc', mc, 'mp', mp, 'l', l, 'g', g, ...
    'Q', Q, 'R', R, 'T', T, 'h', h, ...
    'nx', numel(x0), 'nu', numel(u0));

us = zeros(nu, T/h+1);
Js = zeros(1, T/h+1);
Js_pre = Js;

[t, x] = ode45(@(t,x) SolCart(t,x,u0,setup), 0:h:T, x0);
xs = x';

tol = 1e-6;
max_iter = 200;
iter = 0;
err = inf;
while iter < max_iter && err > tol
    [Ks, ks] = BackwardLQR(xs, us, setup);
    [xs, us, Js] = ForwardLQR(xs, us, Ks, ks, setup);
    
    figure(2), plot(xs(3,:)), hold on
    
    err = norm(Js - Js_pre);
    Js_pre = Js;
    
    fprintf('Iter #%d, (J sum)=%.4f, (J change)=%.4f \n', iter, sum(Js), err)
    iter = iter + 1;
end

showCartPole(t, xs', false)

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

function [Ks, ks] = BackwardLQR(xs, us, setup)
    
    Q = setup.Q;
    R = setup.R;
    T = setup.T;
    h = setup.h;
    nx = setup.nx;
    nu = setup.nu;
    
    C = blkdiag(Q, R);
    V = zeros(size(Q));
    v = zeros(nx, 1);
    
    Ks = zeros(nu, nx, T/h+1);
    ks = zeros(nu, T/h+1);
    
    for i = T/h+1:-1:1
        x = xs(:, i);
        u = us(:, i);
        a = dlarray([x; u]);
        [p_g, pdot_g, th_g, thdot_g] = dlfeval(@(x)CartDynDiff(x,setup), a);
        F = extractdata([p_g, pdot_g, th_g, thdot_g]');
        
        Qf = C + F'*V*F;
        q = blkdiag(Q,R)*[x; u] + F'*v;
        qx = q(1:nx);
        qu = q(nx+1:end);
        Qxx = Qf(1:nx,1:nx);
        Qux = Qf(nx+nu,1:nx);
        Qxu = Qf(1:nx,nx+nu);
        Quu = Qf(nx+nu,nx+nu);
        K = -Quu\Qux;
        k = -Quu\qu;
        V = Qxx + Qxu*K + K'*Qux + K'*Quu*K;
        v = qx + Qxu*k;
        Ks(:,:,i) = K;
        ks(:, i) = k;
    end
end

function [p_g, pdot_g, th_g, thdot_g] = CartDynDiff(x, setup)
    
    p = x(1);
    pdot = x(2);
    th = x(3);
    thdot = x(4);
    u = x(5);
    
    mc = setup.mc;
    mp = setup.mp;
    l = setup.l;
    g = setup.g;
    h = setup.h;
    
    thddot = (g*sin(th)-cos(th)*(u+mp*l*thdot^2*sin(th))/(mc+mp)) / ...
        (l-l*mp*cos(th)^2/(mc+mp));
    pddot = (u+mp*l*(thdot^2*sin(th)-thddot*cos(th)))/(mc+mp);
    
    p_new = p + pdot*h; % + 1/2*pddot*h^2;
    th_new = th + thdot*h; % + 1/2*thddot*h^2;
    pdot_new = pdot + pddot*h;
    thdot_new = thdot + thddot*h;
    
    p_g = dlgradient(p_new,x);
    pdot_g = dlgradient(pdot_new,x);
    th_g = dlgradient(th_new,x);
    thdot_g = dlgradient(thdot_new,x);    
end

function [xs_new, us_new, Js_new] = ForwardLQR(xs, us, Ks, ks, setup)
    
    T = setup.T;
    h = setup.h;
    
    Q = setup.Q;
    R = setup.R;
    
    xs_new = zeros(size(xs));
    us_new = zeros(size(us));
    Js_new = zeros(T/h+1,1);
    
    xs_new(:,1) = xs(:,1);
    ds = zeros([size(xs,1), 1]);

    options = odeset('RelTol',1e-6,'AbsTol',1e-9);
    
    ratios = zeros(T/h+1,1);
    xQxs = zeros(T/h+1,1);
    uRus = zeros(T/h+1,1);
    for i = 1:T/h
        u = us(:,i) + 0.05*(Ks(:,:,i)*ds + ks(:,i));
        x0 = xs(:,i);
        [~, x] = ode45(@(t,x) SolCart(t,x,u,setup), [0, h], x0, options);
        xs_new(:,i+1) = x(end,:)';
%         xs_new(:, i+1) = CartDynDisc(x0, u, setup);
        
        ds = xs_new(:,i+1) - xs(:,i+1);
        us_new(:,i) = u;
        
        Js_new(i) = 1/2*x0'*Q*x0+1/2*u'*R*u;
        
        ratios(i) = (u'*R*u)/(x0'*Q*x0);
        xQxs(i) = 1/2*x0'*Q*x0;
        uRus(i) = 1/2*u'*R*u;
    end
    figure(1), plot(us_new), hold on
    figure(3), plot(xQxs, '-r'), hold on,
    plot(uRus, '-b'), hold on
end

function x_new = CartDynDisc(x, u, setup)
    
    p = x(1);
    pdot = x(2);
    th = x(3);
    thdot = x(4);
    
    mc = setup.mc;
    mp = setup.mp;
    l = setup.l;
    g = setup.g;
    h = setup.h;
    
    thddot = (g*sin(th)-cos(th)*(u+mp*l*thdot^2*sin(th))/(mc+mp)) / ...
        (l-l*mp*cos(th)^2/(mc+mp));
    pddot = (u+mp*l*(thdot^2*sin(th)-thddot*cos(th)))/(mc+mp);
    
    p_new = p + pdot*h; % + 1/2*pddot*h^2;
    th_new = th + thdot*h; % + 1/2*thddot*h^2;
    pdot_new = pdot + pddot*h;
    thdot_new = thdot + thddot*h;
    
    x_new = [p_new, pdot_new, th_new, thdot_new];
end
