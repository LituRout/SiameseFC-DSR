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
p.scaleLR = 0.59; % damping factor for scale update
p.responseUp = 16; % upsampling the small 17x17 response helps with the accuracy
p.windowing = 'cosine'; % to penalize large displacements
p.wInfluence = 0.176; % windowing influence (in convex sum)
p.net = '2016-08-17.net.mat';
%% Additional fine tuning
%center position weight
p.cenWeight = 0.109;%given to previous target position
p.vCorr = 0.107;%given to previous frame
p.sigmaScale = 0.486;%less variance --> more priority is given to bestScale
p.i_angle = -10;
p.f_angle = 10;
p.sweep_angle = 5;%used to generate exemplar features corresponding to rotated z_crop
p.zAngleSweep = 5;%used to reduce number of responseMaps for speed optimization
p.zAngles = p.i_angle:p.sweep_angle:p.f_angle;
p.rMapWeight = 0.95;%higher the rMapWeight higher the weightage given to ground truth map.
p.visualization = true;
p.gpus = 1;
p.bbox_output = false;
p.fout = -1;
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

% -------------------------------------------------------------------------------------------------

sequences = {'vot15_bag',...
    'ball1', ...%%
    'ball2', ...
    'matrix', ...%%
    'gymnastics3', ...
    'bmx', ...
    'crossing', ...
    'butterfly', ...
    'helicopter', ...
    'fish1', ...
    'blanket', ...%%
    'birds2', ...
    'fish3', ...
    'motocross1', ...%%
    'motocross2', ...%%
    'leaves', ...
    'glove', ...%%
    'birds1', ...
    'marching', ...
    'gymnastics2', ...
    'rabbit', ...
    'soccer2', ...
    'hand', ...
    'pedestrian1',...
    'car1'};


for i = 1: numel(sequences)
    tic;
    video = sequences{i};
    fprintf('Running sequence : %s\n',video)
    tracker_all_dataSets(p,video);
    toc;
end