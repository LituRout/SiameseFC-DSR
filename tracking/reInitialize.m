function [targetPosition,targetSize]= reInitialize(im)

close all;
figure(1), imshow(uint8(im),[]);
%     figure(1), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');

r = getrect;
%      	[cx, cy, w, h] = getrect;
%     close all;
targetPosition = [r(2)+r(4)/2 r(1)+r(3)/2]; % centre of the bounding box
targetSize = [r(4) r(3)];
fprintf('\nNew Target Position : %f %f New Target Size %f %f\n',targetPosition(1),targetPosition(1),targetSize(1),targetSize(2));
%     h = rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
drawnow
close
end