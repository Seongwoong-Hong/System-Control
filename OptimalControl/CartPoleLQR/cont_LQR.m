close all

mc = 5;
mp = 1;
l = 0.5;
g = 9.81;


A = [0, 1, 0, 0;
    -mc/mp*g, 0, 0, 0;
    0, 0, 0, 1;
    0, 0, -(mc+mp)*g/(mc*l), 0];
B = [0; 1/mc; 0; 1/(mc*l)];
Q = diag([0, 1, 10, 10]);
R = 0.1;
N = 0;
G = zeros(4,4);

T = 5;
h = 0.05;

x0 = [0, 0, 0.1, 0];
[t, x] = ode45(@(t,x) contCartPole(t,x,mc,mp,l,T,A,B,Q,R,G), 0:h:T, x0);

figure, plot(t, x(:,1)), title('x')
figure, plot(t, x(:,3)), title('theta')

showCartPole(t, x, true)

[~, Ks] = ode45(@(t,X)mRiccati(t,X,A,B,Q,R), 0:h:T, G(:));
Ks = flipud(Ks);

V = zeros(numel(t), 1);
for i = 1:numel(t)
   K = reshape(Ks(i,:), size(A));
   V(i) = x(i,:)*K*x(i,:)';
end

U = zeros(numel(t),1);
for i = numel(t):-1:1
    if i == numel(t)
        U(i) = x(i,:)*Q*x(i,:)';
    else
        u = -R\B'*K*x(i,:)';
        U(i) = U(i+1) + x(i,:)*Q*x(i,:)' + u'*R*u;
    end
end

figure, semilogy(t, V)
hold on, semilogy(t, U), title('Value')


function dxdt = contCartPole(t,x,mc,mp,l,T,A,B,Q,R,G)
%     p = x(1);
    pdot = x(2);
    th = x(3);
    thdot = x(4);
    
    g = 9.81;
    
    % H*qddot + C*qdot + E = D*u
    H = [mc+mp, mp*l*cos(th); cos(th), l];
    C = [0, -mp*l*thdot*sin(th); 0, 0];
    E = [0; g*sin(th)];
    D = [1; 0];
    
%     K = lqr(A, B, Q, R, zeros(size(x)));
%     u = -K*x;
    K = mLQR(t, T, A, B, Q, R, G);
    u = -R\B'*K*x;
    
    dotdot = -H\C*[pdot; thdot] - H\E + H\D*u;
    
    dxdt = [pdot; dotdot(1); thdot; dotdot(2)];
end

function K = mLQR(t, T, A, B, Q, R, G)
    if t >= T
        K = zeros(size(A));
    else
        [~, Ks] = ode45(@(t,X)mRiccati(t,X,A,B,Q,R), [0, T-t], G(:));
        K = reshape(Ks(end,:), size(A));
    end
end

function dXdt = mRiccati(~, X, A, B, Q, R)
    X = reshape(X, size(A));
    dXdt = - A'.*X - X*A + X*B*R^(-1)*B'*X - Q;
    dXdt = dXdt(:);
end
