function X = collect_data(d, x_min, x_max, method)

    nx = numel(x_min);
    X = zeros(nx,d);
     
    if strcmp(method, 'rnd')
    
        threshold = 1e-3;

        idx = 1;
        while idx <= d

            x_candidate = (x_max - x_min).*rand(nx,1) + x_min;

            if min(dist(X,x_candidate')) >= threshold
                X(:,idx) = x_candidate;
                idx = idx + 1;
                if mod(d,100) == 0, disp([num2str(idx) ' samples collected...']); end
            end

        end
        
        X = X';
    
    elseif strcmp(method, 'grid')
    
        d_dim = ceil(d^(1/(nx)));
        aux = linspace(0,1,d_dim);
        X_aux = x_min + aux.*(x_max - x_min);
        
        for idx = 1:nx
            coords{idx} = X_aux(idx,:);
        end
        data = cell(1,nx);
        [data{:}] = ndgrid(coords{:});
        X = [];
        for i = 1:nx
            X = [X data{i}(:)];
        end   
        
        % drop rnd elements to return exactly D elements
        rng(1);
        for i = 1:(size(X,1)-d)
            idx = randi([1,size(X,1)]);
            X(idx,:) = [];
        end
     
    end

end