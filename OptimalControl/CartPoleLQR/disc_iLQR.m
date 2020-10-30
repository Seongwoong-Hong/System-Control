close all

T  = 5;
h = 0.005;

mc = 5;
mp = 1;
l = 0.5;
g = 9.81;
x0 = [0; 0; -pi*1/2; 0];
% u0 = 0;
u0 = us(1);

Q = diag([1, 1, 1, 1]);
R = 0.001;

nx = numel(x0);
nu = numel(u0);

setup = struct('mc', mc, 'mp', mp, 'l', l, 'g', g, ...
    'Q', Q, 'R', R, 'T', T, 'h', h, 'J0_max', 1e8, ...
    'nx', numel(x0), 'nu', numel(u0), ...
    'b_low', 1e-8, 'b_high', 10, 'a_decay', 0.5, 'line_iter_max', 10, ...
    'reg_iter_max', 5);

% us = zeros(nu, T/h+1);
J0_pre = inf;

% [t, x] = ode45(@(t,x) SolCart(t,x,u0,setup), 0:h:T, x0);
% xs = x';
% xs = repmat(x0, 1, T/h+1);

reg_iter = 0;

err = inf;
rou = 1e-8;
rou_low = 1e-8;
rou_high = 1e-6;
rou_scale = 1.6;
J_max = 1e8;

tol = 1e-2;
max_iter = inf;
iter = 0;
while iter < max_iter && err > tol
    xs_old = xs;
    us_old = us;
    
    [Ks, ks, fV] = BackwardLQR(xs, us, rou, setup);
    [xs, us, J0, suc] = ForwardLQR(xs, us, Ks, ks, fV, setup);
    
    if ~suc
        fprintf('\t Enhance regularization %.2E \n', rou)
        rou = rou*rou_scale;
        assert(rou < rou_high, 'Value blows up / too many failures')
        if reg_iter < setup.reg_iter_max
            xs = xs_old;    % discard forward path
            us = us_old;
            reg_iter = reg_iter + 1;
            continue
        else
            reg_iter = 0;
        end
    elseif rou > rou_low
        fprintf('\t Release regularization %.2E \n', rou)
        rou = rou/rou_scale;
    end
        
    err = norm(J0 - J0_pre);
    J0_pre = J0;
    iter = iter + 1;
    
    fprintf('Iter #%d, (J sum)=%.4f, (J change)=%.4f \n', iter, sum(J0), err)
    stackFig(1, xs(3,:), 'theta')
end

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

function [Ks, ks, fV] = BackwardLQR(xs, us, rou, setup)
    
    Q = setup.Q;
    R = setup.R;
    T = setup.T;
    h = setup.h;
    nx = setup.nx;
    nu = setup.nu;
    
    C = blkdiag(Q, R);
    V = zeros(size(Q));
    v = zeros(nx, 1);
    
    Quus = zeros(nu, nu, T/h+1);
    qus = zeros(nu, T/h+1);
    
    Ks = zeros(nu, nx, T/h+1);
    ks = zeros(nu, T/h+1);
    
    for i = T/h+1:-1:1
        x = xs(:, i);
        u = us(:, i);
        a = dlarray([x; u]);
        [p_g, pdot_g, th_g, thdot_g] = dlfeval(@(x)CartDynDiff(x,setup), a);
        F = extractdata([p_g, pdot_g, th_g, thdot_g]');
                
        Qf = C + F'*V*F;
        q = C*[x; u] + F'*v;
        qx = q(1:nx);
        qu = q(nx+1:end);
        Qxx = Qf(1:nx,1:nx);
        Qux = Qf(nx+nu,1:nx);
        Qxu = Qux';
        Quu = Qf(nx+nu,nx+nu);
        K = -(Quu+rou*eye(nu))\Qux;     % regularization
        k = -(Quu+rou*eye(nu))\qu;
        V = Qxx + Qxu*K + K'*Qux + K'*Quu*K;
        v = qx + K'*Quu*k + K'*qu + Qxu*k;
        Ks(:,:,i) = K;
        ks(:, i) = k;
        
        Quus(:,:,i) = Quu;
        qus(:,i) = qu;
    end
    blks = squeeze(mat2cell(Quus,nu,nu,ones(1,T/h+1)));
    c1 = ks(:)'*qus(:);
    c2 = ks(:)'*blkdiag(blks{:})*ks(:);
    fV = @(a) a*c1 + 0.5*a^2*c2;
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
    
    p_new = p + pdot*h;
    th_new = th + thdot*h;
    pdot_new = pdot + pddot*h;
    thdot_new = thdot + thddot*h;
    
    p_g = dlgradient(p_new,x);
    pdot_g = dlgradient(pdot_new,x);
    th_g = dlgradient(th_new,x);
    thdot_g = dlgradient(thdot_new,x);    
end

function [xs_new, us_new, J0_new, success] = ...
    ForwardLQR(xs, us, Ks, ks, fV, setup)
    
    T = setup.T;
    h = setup.h;
    Q = setup.Q;
    R = setup.R;
    
    b_low = setup.b_low;
    b_high = setup.b_high;
    a_decay = setup.a_decay;
    iter_max = setup.line_iter_max;
    
    xs_new = zeros(size(xs));
    us_new = zeros(size(us));
    J0_max = setup.J0_max;
    xs_new(:,1) = xs(:,1);
    
    blks_Q = repmat({Q},1,T/h+1);
    blks_R = repmat({R},1,T/h+1);
    J0_old = 1/2*xs(:)'*blkdiag(blks_Q{:})*xs(:) +...
        1/2*us(:)'*blkdiag(blks_R{:})*us(:);
       
    xQxs = zeros(T/h+1,1);
    uRus = zeros(T/h+1,1);
    
    options = odeset('RelTol',1e-6,'AbsTol',1e-9);
    success = true;
    a = 1;
    line_iter = 0;
    while line_iter < iter_max
        J0_new = 0;
        dx = zeros([size(xs,1), 1]);
        
        for i = 1:T/h
            u = us(:,i) + Ks(:,:,i)*dx + a*ks(:,i);
            x0 = xs_new(:,i);
            [~, x] = ode45(@(t,x) SolCart(t,x,u,setup), [0, h], x0, options);
            xs_new(:,i+1) = x(end,:)';
            us_new(:,i) = u;

            dx = xs_new(:,i+1) - xs(:,i+1);

            J0_new = J0_new + 1/2*x0'*Q*x0 + 1/2*u'*R*u;
            xQxs(i) = 1/2*x0'*Q*x0;
            uRus(i) = 1/2*u'*R*u;
            
            if J0_new > J0_max
                success = false;
                return
            end
        end
        inc_ratio = (J0_old - J0_new)/(-fV(a));
        if inc_ratio > b_low && inc_ratio < b_high
            break
        end
        
        a = a_decay*a;
        line_iter = line_iter + 1;
    end
    
    stackFig(2, xQxs, 'x J')
    stackFig(3, uRus, 'u J')
    
    if line_iter == iter_max
        success = false;
        return
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