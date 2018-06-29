% function run_experiment()
%RUN_EXPERIMENT
    global imdb_video;
%	contains the parameters that are specific to the current experiment.
% 	These are the parameters that change frequently and should not be committed to
% 	the repository but should be saved with the results of the experiment.
    imdb_video = load( 'C:\AI_ML\Matlab\siamese-fc-master\imdb_video.mat');
	% Parameters that should have no effect on the result.
	opts.prefetch = false;
	opts.gpus = 1;

	% Parameters that should be recorded.
	% opts.foo = 'bla';
% 
% 	if nargin < 1
% 	    imdb_video = [];
% 	end
    imdb_video_ = imdb_video.imdb_video;
	experiment(imdb_video_, opts);

% end
% run_experiment('C:\AI_ML\Matlab\siamese-fc-master\imdb_video.mat');
