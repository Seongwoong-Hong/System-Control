function showCartPole(t, x, gifSave)

figure('Name','Bipedal Model Simulation',...
    'NumberTitle','off',...
    'Menubar', 'none',...
    'Toolbar', 'none',...
    'Color', 'w',...
    'Position', [300, 300, 500, 500]);

axis equal
% axis off, axis equal

xlim([-10, 10]);
ylim([0, 10]);
yOff = 2;

patch([-10; 10; 10; -10], yOff + 0.1*[-1; -1; 0.1; 0.1], ...
    'black', 'EdgeColor', 'none');

h = 3*(t(2)-t(1));

l = 7;
p = x(:, 1);
th = x(:, 3);

timeTxt = text(-10, 15, 'Time elapse: 0.00 sec','FontSize',15);

wCart = 2.0;
hCart = 1.6;
Cart = patch(wCart/2*[-1, 1, 1, -1], ...
    yOff + hCart/2*[-1, -1, 1, 1], ...
    'blue', 'EdgeColor', 'none');
Pole = line(p(1)+[0, l*sin(th(1))], yOff+[0, l*cos(th(1))],...
    'LineWidth', 3, 'Color', 'g', 'Marker', 'o', 'MarkerFaceColor', 'g', ...
    'MarkerEdgeColor', 'none', 'MarkerSize', 10);

for i = 2:numel(th)
    % Save animation as gif file
    if gifSave
        frame = getframe(gcf);
        img = frame2im(frame);
        if i == 2
            [imind, cm] = rgb2ind(img, 128);
            imwrite(imind,cm,'carPoleLQR.gif','gif','Loopcount',Inf,'DelayTime', h);
        else
            imind = rgb2ind(img, cm);
            imwrite(imind,cm,'carPoleLQR.gif','gif','WriteMode','append','DelayTime',h);
        end
    end
    
    pause(h);
    Cart.XData = p(i) + wCart/2*[-1, 1, 1, -1];
    
    Pole.XData = p(i) + [0, l*sin(th(i))];
    Pole.YData = yOff + [0, l*cos(th(i))];
    
    timeTxt.String = sprintf('Time elapse: %.2f sec',t(i));
end
