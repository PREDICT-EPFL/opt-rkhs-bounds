function min_queries_dists(datasets, x, u)
% Study the distance of the dataset and the query points

    N = size(datasets,2);

    for step = 1:N

        z = [x(:,1)' u(1:step)];
        Z = datasets{1,step}(:,1:end-1);
        min_dists(step) = min(dist(Z,z'));

    end

    disp('Distance between query point and the closest data point, for every step:')
    min_dists

end

