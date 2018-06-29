% -------------------------------------------------------------------------------------------------------------------------
function [newTargetPosition, bestScale,newAngle,maxScore] = tracker_eval(net_x, s_x, scoreId, z_features_r, x_crops, targetPosition, window, p,angle)
% function [newTargetPosition, bestScale] = tracker_eval(net_x, s_x, scoreId, z_features, x_crops, targetPosition, window, p)

%TRACKER_STEP
%   runs a forward pass of the search-region branch of the pre-trained Fully-Convolutional Siamese,
%   reusing the features of the exemplar z computed at the first frame.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
% forward pass, using the pyramid of scaled crops as a "batch"
%%%litu
scaleBins = 1:p.numScale;
bins = ((p.f_angle-p.i_angle)/p.sweep_ang) + 1;
responseMap_r = gpuArray(single(zeros(272,272,bins)));
responseScore = gpuArray(zeros(bins,1));

for i = 1:bins
    z_features = z_features_r(:,:,:,:,i);
    net_x.eval({p.id_feat_z, z_features, 'instance', x_crops});
    responseMaps = reshape(net_x.vars(scoreId).value, [p.scoreSize p.scoreSize p.numScale]);
    responseMapsUP = gpuArray(single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, p.numScale)));
    %     responseMapsUP = (single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, p.numScale)));
    
    % Choose the scale whose response map has the highest peak
    if p.numScale>1
        currentScaleID = ceil(p.numScale/2);
        bestScale = currentScaleID;
        bestPeak = -Inf;
        for s=1:p.numScale
            if p.responseUp > 1
                % upsample to improve accuracy
                responseMapsUP(:,:,s) = imresize(responseMaps(:,:,s), p.responseUp, 'bicubic');
            else
                responseMapsUP(:,:,s) = responseMaps(:,:,s);
            end
            thisResponse = responseMapsUP(:,:,s);
            % penalize change of scale
            if s~=currentScaleID, thisResponse = thisResponse * p.scalePenalty; end
            thisPeak = max(thisResponse(:));
            if thisPeak > bestPeak
                bestPeak = thisPeak;
                bestScale = s;
            end
        end
        %% responseMap Scale correction(more weight is given to best scale)
        mu1 = bestScale;
        scaleWeights = exp(-((scaleBins - mu1)./p.sigmaScale).^2);
        responseMap_ = gpuArray(single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp)));
        for k = 1:p.numScale
            responseMap_ = responseMap_ + responseMapsUP(:,:,k) * scaleWeights(k);
        end
        responseMap = responseMap_./p.numScale;
        %%
%         responseMap = responseMapsUP(:,:,bestScale);
    else
        responseMap = responseMapsUP;
        bestScale = 1;
    end
    responseMap_r(:,:,i) = responseMap;
    
    responseScore(i,1) = max(max(responseMap));
    
end


  %% compute weighted average responseMap and weighted average newAngle
    x = 1:bins;
    % sigma = 1.27;%better performance
    sigma = p.sigmaRMap;
    m = p.m;%top m maxScore
    newAngle = gpuArray(zeros(m,1));
    newTargetPosition = gpuArray(zeros(m,2));
    maxScore = gpuArray(zeros(m,1));
for j = 1:m  
    
    % find response map corresponding to highest score and take the index
    maxScoreIndex = find(responseScore == max(responseScore));
    maxScoreIndex = maxScoreIndex(1);
    maxScore(j) = responseScore(maxScoreIndex);
    % replace max response Score with minimum responseScore so that in the
    % next iteration next maxima gets selected.
    responseScore(maxScoreIndex) = min(responseScore);
    
    %Gaussian weighing coefficients centered at maxScoreIndex
    mu = maxScoreIndex;
    % gaussian weights centered at maxScoreIndex
    weights = exp(-((x-mu)/sigma).^2);
    responseMap_avg = gpuArray(zeros(size(responseMap)));
    
    for k = 1:bins
        responseMap_avg = responseMap_avg + (responseMap_r(:,:,k).*weights(k));
    end
    responseMap_avg = responseMap_avg./bins  ;% average of all the responseMaps.
    responseMap = responseMap_avg;
    
    %compute weighted average angle (highest weight to previous angle)
    %maximum weight is given to previous angle
    MapAngle= p.i_angle + p.sweep_ang*(maxScoreIndex-1);
    newAngle(j) = round(MapAngle*(1- p.angWeight)+angle*p.angWeight);
%     fprintf('newAngle : %d\n',newAngle); 
    
    %%    
    % make the response map sum to 1
    responseMap = responseMap - min(responseMap(:));
    responseMap = responseMap / sum(responseMap(:));
    % apply windowing
    responseMap = (1-p.wInfluence)*responseMap + p.wInfluence*window;
    [r_max, c_max] = find(responseMap == max(responseMap(:)), 1);
    [r_max, c_max] = avoid_empty_position(r_max, c_max, p);
    p_corr = [r_max, c_max];
    % Convert to crop-relative coordinates to frame coordinates
    % displacement from the center in instance final representation ...
    disp_instanceFinal = p_corr - ceil(p.scoreSize*p.responseUp/2);
    % ... in instance input ...
    disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
    % ... in instance original crop (in frame coordinates)
    disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
    % position within frame in frame coordinates
    newTargetPosition(j,:) = targetPosition + disp_instanceFrame;
end

end

function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
if isempty(r_max)
    r_max = ceil(params.scoreSize/2);
end
if isempty(c_max)
    c_max = ceil(params.scoreSize/2);
end
end
