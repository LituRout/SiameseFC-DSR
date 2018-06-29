function [x_new,y_new] = rotate_rectangle(x,y,angle)
% x = [1 2 2 1];
% y = [1 1 2  2];

w = x(2)- x(1);
h = y(4) - y(1);
xr = x(1) + w/2 ;
yr = y(1) + h/2 ;

% bbox_rot = [cosd(angle) -sind(angle) xr*(1-cosd(angle))+yr*sind(angle);
%     sind(angle) cos(angle) yr*(1-cosd(angle))-xr*sind(angle);
%     0 0 1];
% translate = [1 0 -xr; 0 1 -yr;0 0 1];
%% translate
x_t = x - xr;
y_t = y - yr;
%% new x,y
x = x_t;
y =y_t;
befRotTarget = [x;y];
%% rotate
bbox_rot = [cosd(angle) sind(angle) ;
    -sind(angle) cos(angle)];
newCoord = bbox_rot * befRotTarget ;
x_new = newCoord(1,:);
y_new = newCoord(2,:);

%% translate back to xr, yr
x_new = x_new + xr;
y_new = y_new + yr;
end

