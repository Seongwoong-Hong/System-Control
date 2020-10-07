
global mc mp l g h

mc = 5;
mp = 1;
l = 0.5;
g = 9.81;

h = 0.05;

x0 = dlarray([1,2,3,4,5]);
p = 0;
[th_g, thdot_g] = dlfeval(@(x)foo(x,p),x0);

function [th_g, thdot_g] = foo(x,param)
    global mc mp l g h
    
    p = x(1);
    pdot = x(2);
    th = x(3);
    thdot = x(4);
    u = x(5);
    
    thddot = (g*sin(th)-cos(th)*(u+mp*l*thdot^2*sin(th))/(mc+mp)) / ...
        (l+l*mp*cos(th)^2/(mc+mp));
    pddot = (u+mp*l*(thdot^2*sin(th)-thddot*cos(th)))/(mc+mp);
    
    p_new = p + pdot*h + 1/2*pddot*h^2;
    th_new = th + thdot*h + 1/2*thddot*h^2;
    pdot_new = pdot + pddot*h;
    thdot_new = thdot + thddot*h;
%     
%     p_g = dlgradient(p_new,x,'RetainData', true);
%     pdot_g = dlgradient(pdot_new,x,'RetainData', true);
    th_g = dlgradient(th_new,x,'RetainData', true);
    thdot_g = dlgradient(thdot_new,x,'RetainData', true);
    
end