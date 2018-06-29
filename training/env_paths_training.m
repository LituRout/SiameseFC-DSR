function opts = env_paths_training(opts)

    opts.rootDataDir = 'C:\AI_ML\Matlab\DataSets\ILSVRC2015\Data\VID\train\';
%     opts.imdbVideoPath = 'path/to/imdb_video.mat';
    
    opts.imdbVideoPath = 'C:\AI_ML\Matlab\siamese-fc-master\imdb_video.mat';
%     opts.imageStatsPath = '/path/to/imageStats.mat';
    opts.imageStatsPath = 'C:\AI_ML\Matlab\siamese-fc-master\ILSVRC2015.stats.mat';
    
end
