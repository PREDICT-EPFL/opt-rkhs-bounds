function min_features_dists(datasets)
% Study the distances among dataset features

    N = size(datasets,2);

    min_dists = inf(1,N);
    for step = 1:N
        
        data = datasets{1,step}(:,1:end-1);
        d = size(data,1);
        
        for i = 1:d
    
            data_temp = data;
            sample = data_temp(i,:);
            data_temp(i,:) = [];
            sample_min_dist = min(dist(data_temp,sample'));
            if sample_min_dist < min_dists(step), min_dists(step) = sample_min_dist; end
        
        end
        
    end
    
    disp('The minimum distances for every dataset:')
    min_dists

end

