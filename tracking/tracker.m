% -------------------------------------------------------------------------------------------------
%All rotated response Maps
%Tracking with modified fully functional rotated bounding box compatible with top m(m=1) maxScoreIndex
%displacement correction
%angle of bbox correction(no rotated bbox)
%ratio of max score and dist correction
%velocity correction
%
% -------------------------------------------------------------------------------------------------
function bboxes = tracker(video,varargin)
% startup
%TRACKER
%   is the main function that performs the tracking loop
%   Default parameters are overwritten by VARARGIN
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------
% These are the default hyper-params for SiamFC-3S
% The ones for SiamFC (5 scales) are in params-5s.txt
global p;
p.numScale = 3;
p.scaleStep = 1.0375;
p.scalePenalty = 0.9745;
% p.scaleLR = 0.59; % damping factor for scale update
p.scaleLR = 0.59; % works better butterfly
p.responseUp = 16; % upsampling the small 17x17 response helps with the accuracy
p.windowing = 'cosine'; % to penalize large displacements
% p.wInfluence = 0.176; % windowing influence (in convex sum)
p.wInfluence = 0.2;%% better 10
p.net = '2016-08-17.net.mat';
%   p.net = '2016-08-17gray025.net.mat';
%% additional hyper parameters
p.cenWeight = 0.202;%%given to previous targetPosition
p.vCorr = 0.105;%given to previous angular direction

p.sigmaScale = 0.4;%%given to best scale
p.m = 3;% top m avg responseMaps
p.sigmaRMap = 1.273;%%std of maxScoreResponseMap
p.angWeight = 0.7;%given to previous angle of bbox
%% rotated z_crop initialization
p.f_angle = 60;
p.i_angle = -60;
p.sweep_ang = 10;
%% execution, visualization, benchmark
% p.video = 'vot15_bag';
    p.video = 'basketball';
%     p.video = 'ball1';%%
% p.video = 'ball2';
%     p.video = 'matrix';%%
%     p.video = 'gymnastics3';
%       p.video = 'bmx';
%       p.video = 'crossing';
% p.video = 'butterfly';
%       p.video = 'helicopter';
% p.video = 'fish1';
% p.video = 'blanket';%%
% p.video = 'birds2';
% p.video = 'fish3';
% p.video = 'fish2';
% p.video = 'handball2';
%  p.video = 'motocross1';%%
% p.video = 'motocross2';%%
%  p.video = 'leaves';
%  p.video = 'glove';%%
% p.video = 'birds1';
%  p.video = 'marching';
% p.video = 'gymnastics2';
% p.video = 'rabbit';
%  p.video = 'soccer2';
%  p.video = 'hand';
%  p.video = 'pedestrian1';
% p.video = 'singer1';
% p.video = 'octopus';
% p.video = 'car2';

%%%
% p.video = video;
%%%
p.savePath = ['C:\AI_ML\Matlab\siamese-fc-master\siameseResults\' p.video '\'];
if ~exist(p.savePath,'file')
    mkdir(p.savePath);
end
%%
p.visualization = true;
p.gpus = 1;
p.bbox_output = false;
p.fout = -1;
%% Params from the network architecture, have to be consistent with the training
p.exemplarSize = 127;  % input z size
p.instanceSize = 255;  % input x size (search region)
p.scoreSize = 17;
p.totalStride = 8;
p.contextAmount = 0.5; % context amount for the exemplar
p.subMean = false;
%% SiamFC prefix and ids
p.prefix_z = 'a_'; % used to identify the layers of the exemplar
p.prefix_x = 'b_'; % used to identify the layers of the instance
p.prefix_join = 'xcorr';
p.prefix_adj = 'adjust';
p.id_feat_z = 'a_feat';
p.id_score = 'score';
% Overwrite default parameters with varargin
p = vl_argparse(p, varargin);
% -------------------------------------------------------------------------------------------------

% Get environment-specific default paths.
p = env_paths_tracking(p);
% Load ImageNet Video statistics
if exist(p.stats_path,'file')
    stats = load(p.stats_path);
else
    %     warning('No stats found at %s', p.stats_path);
    stats = [];
end

% Load two copies of the pre-trained network
net_z = load_pretrained([p.net_base_path p.net], p.gpus);
net_x = load_pretrained([p.net_base_path p.net], []);
[imgFiles, targetPosition, targetSize] = load_video_info(p.seq_base_path, p.video);
nImgs = numel(imgFiles);
startFrame = 1;
% Divide the net in 2
% exemplar branch (used only once per video) computes features for the target
remove_layers_from_prefix(net_z, p.prefix_x);
remove_layers_from_prefix(net_z, p.prefix_join);
remove_layers_from_prefix(net_z, p.prefix_adj);
% instance branch computes features for search region x and cross-correlates with z features
remove_layers_from_prefix(net_x, p.prefix_z);
zFeatId = net_z.getVarIndex(p.id_feat_z);
scoreId = net_x.getVarIndex(p.id_score);
% get the first frame of the video
im = gpuArray(single(imgFiles{startFrame}));
% if grayscale repeat one channel to match filters size
if(size(im, 3)==1)
    im = repmat(im, [1 1 3]);
end
% Init visualization
videoPlayer = [];
if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
    videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
end
% get avg for padding
avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);

wc_z = targetSize(2) + p.contextAmount*sum(targetSize);
hc_z = targetSize(1) + p.contextAmount*sum(targetSize);
s_z = sqrt(wc_z*hc_z);
scale_z = p.exemplarSize / s_z;
% initialize the exemplar
[z_crop, ~] = get_subwindow_tracking(im, targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans);
if p.subMean
    z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
end

d_search = (p.instanceSize - p.exemplarSize)/2;
pad = d_search/scale_z;
s_x = s_z + 2*pad;
% arbitrary scale saturation
min_s_x = 0.2*s_x;
max_s_x = 5*s_x;

switch p.windowing
    case 'cosine'
        window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
    case 'uniform'
        window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
end
% make the window sum 1
window = window / sum(window(:));
scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));
% evaluate the offline-trained network for exemplar z features
%%
bins = ((p.f_angle - p.i_angle)/p.sweep_ang) + 1;
z_features = gpuArray(zeros(6,6,256,p.numScale,bins));%[6 6 256] is the feature dimension
j=1;
z_crop_org = z_crop;
for i = p.i_angle:p.sweep_ang:p.f_angle
    z_crop_r = imrotate(z_crop_org,i,'crop');
    z_crop_r = imresize(z_crop_r,[p.exemplarSize p.exemplarSize]);
    net_z.eval({'exemplar', z_crop_r});
    z_features_r = net_z.vars(zFeatId).value;
    z_features_r = repmat(z_features_r, [1 1 1 p.numScale]);
    z_features(:,:,:,:,j) = z_features_r;
    j = j+1;
end
z_features = single(z_features);
%%
bboxes = zeros(nImgs, 4);
% start tracking
newAngle = 0;
theta1 = 0;
dVector0 = 0;
tic;
for i = startFrame:nImgs
    if i>startFrame
        % load new frame on GPU
        im = gpuArray(single(imgFiles{i}));
        % if grayscale repeat one channel to match filters size
        if(size(im, 3)==1)
            im = repmat(im, [1 1 3]);
        end
        scaledInstance = s_x .* scales;
        scaledTarget = [targetSize(1) .* scales; targetSize(2) .* scales];
        % extract scaled crops for search region x at previous target position
        x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
        %% make sure that bbox coordinates are positive always i.e. within
        % the frame.
        targetPosition = abs(targetPosition);
        %% evaluate the offline-trained network for exemplar x features
        angle = newAngle;
        [newTargetPosition, newScale,newAngle,maxScore] = tracker_eval(net_x, round(s_x), scoreId, z_features, x_crops, targetPosition, window, p,angle);
        
       
        %% now all the newTargetPositions are good candidates.which one to choose?
        % maximum (Score/Distance).distance measured from previous
        % targetPosition
        dist_ = newTargetPosition - targetPosition;
        dist = gpuArray(zeros(p.m,1));
        for l = 1:p.m
            dist(l) = norm(dist_(l,:));
        end
        ratioScDist = maxScore./dist;
        bestPosIndex = find(ratioScDist == max(ratioScDist));% best possible index
        newTargetPosition = newTargetPosition(bestPosIndex,:);
        % in case of multiple bestPosIndex take the first one.
        newTargetPosition = newTargetPosition(1,:);
        %% Rotation angle of bbox (which corresponds to z_crop) after correction in tracker_eval
        newAngle = newAngle(bestPosIndex);
        % in case of multiple bestPosIndex take the first one.
        newAngle = newAngle(1);
        % newAngle = 0;
        
        %% Dynamics(Velocity) correction
        % center of newTargetPosition
        c_x2 = newTargetPosition(2);
        c_y2 = newTargetPosition(1);
        % center of targetPosition(previous)%coloumn --> x%row --> y
        c_x1 = targetPosition(2);
        c_y1 = targetPosition(1);
        % compute angular displacement
        theta2 = atan2d((c_y2 - c_y1),(c_x2 - c_x1));
        % p.vCorr is velocity correction.
        theta2 =(1 - p.vCorr)*theta2 + p.vCorr*theta1;
        theta1 = theta2;
        
        %displacement Vector
        dVector = abs(sqrt((c_x2-c_x1)^2 + (c_y2-c_y1)^2));
         %% displacement correction
        dVector = p.cenWeight * dVector0 + (1-p.cenWeight) * dVector;
        dVector0 = dVector;
        % recompute newTargetPosition
        newTargetPosition(2) = targetPosition(2) + dVector*cosd(theta2);
        newTargetPosition(1) = targetPosition(1) + dVector*sind(theta2);
        % make sure that bbox does not go outside the frame
        newTargetPosition = abs(newTargetPosition);
        %%
        targetPosition = gather(newTargetPosition);
        %% targetPosition of maxScoreIndex should be given more priority compared to bestPosIndex
        %         bestScWt = 0.9;
        %         targetPosition = bestScWt * targetPosition + (1-bestScWt)*targetPosition1;       
        
        %% scale damping and saturation
        s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
        targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
        
    end
    %% output bbox in the original frame coordinates
    oTargetPosition = targetPosition; % .* frameSize ./ newFrameSize;
    oTargetSize = targetSize; % .* frameSize ./ newFrameSize;
    bboxes(i, :) = [oTargetPosition([2,1]) - oTargetSize([2,1])/2, oTargetSize([2,1])];
    % make sure that bbox coordinates are positive always i.e. within
    % the frame. This increased performance drastically.
    bboxes(i,:) = abs(bboxes(i,:));
    %% get rotated bounding box
    newCoord = rotate_bbox(oTargetPosition,oTargetSize,newAngle);
    %% get axis alligned bbox from rotated polygon with the same
    % area as polygon.(cx --> center x)
    [cx, cy, w, h] = get_axis_aligned_BB(newCoord);
    bbox = gather(abs([cx-w/2, cy-h/2, w, h]));
    %%
    if p.visualization
        if isempty(videoPlayer)
            figure(1), imshow(im/255);
            figure(1), rectangle('Position', bbox, 'LineWidth', 2, 'EdgeColor', 'r');
            drawnow
            fprintf('Frame %d\n', startFrame+i);
        else
            im = gather(im)/255;
            im = insertShape(im,'Polygon',gather(newCoord),'LineWidth',4,'Color','red');
%             im = insertShape(im, 'Rectangle', bbox, 'LineWidth', 2, 'Color', 'red');
            %% write output sequences
            imwrite(im,[p.savePath num2str(i) '.jpg']);
            % Display the annotated video frame using the video player object.
            step(videoPlayer, im);
        end
    end
end
end
