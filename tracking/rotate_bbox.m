function [newCoord] = rotate_bbox(targetPosition,targetSize,angle)
% targetPosition = [250 200];
% targetSize = [100 200];
xr = targetPosition(2);
yr = targetPosition(1);
w = targetSize(2);
h = targetSize(1);
coord1 = [xr-w/2 yr-h/2 xr+w/2 yr-h/2 xr+w/2 yr+h/2 xr-w/2 yr+h/2];
% im = imread('peppers.png');
% im1 = insertShape(rgb2gray(im),'Polygon',coord1,'color','red');
% imshow(im1);
%%
x = coord1(1:2:end);
y = coord1(2:2:end);
%% translate to origin
x_t = x - xr;
y_t = y - yr;
%% new x,y
x = x_t;
y =y_t;
befRotTarget = [x;y];
%% rotate
% angle = 45;
bbox_rot = [cosd(angle) sind(angle) ;
    -sind(angle) cosd(angle)];
newCoord = bbox_rot * befRotTarget ;
x_new = newCoord(1,:);
y_new = newCoord(2,:);

%% translate back to xr, yr
x_new = x_new + xr;
y_new = y_new + yr;
coord2 = [x_new(1) y_new(1) x_new(2) y_new(2) x_new(3) y_new(3) x_new(4) y_new(4)];
newCoord = coord2;
% im1 = insertShape(rgb2gray(im),'Polygon',coord2,'color','red');
% figure(2);
% imshow(im1);
end

