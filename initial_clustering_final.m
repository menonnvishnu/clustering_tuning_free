function [points_list,initial_cluster] = initial_clustering_final(X)
%%
%Author: Vishnu
%This Function takes as input the data matrix and returns initial clustering
%with minimum of 3 data points in each constituent cluster, also returns the cluster index map for each point    
%%
%Calculating Angles
    [~,N] = size(X);
    C1 = abs(acos(abs(X'*X)));    
    for i=1:N
        C1(i,i) = 10;
    end
%%    
%Forming the relationship matrix
    r = cell(1,N);
    for i=1:N
        [~,r{1,i}] = sort(C1(i,:));
    end
% r contains in each cell the order of correlations with other points 
% for each data point
%%
%Clustering starts
    already_taken = [];%for keeping track of already classified points
    count = 0;
    points_list = zeros(1,N);%Points list has the cluster index of each data point - intialized to zero
    initial_cluster = cell(1,1);
    
    for i=1:N 
        %Take the relationship and form a new cluster with top 2 and the point itself if the point is not already taken
        A = r{1,i};
        if isempty(find(already_taken==i, 1)) && isempty(find(already_taken==A(1), 1)) && isempty(find(already_taken==A(2), 1))
            count = count+1;
            already_taken = [already_taken,i,A(1),A(2)];
            initial_cluster{count} = [i,A(1),A(2)];
            points_list(initial_cluster{count}) = count;
        end
    end
    %Take all the not classifed points and add it to the cluster which
    %contains its nearest point in terms of angle
    rem_p = find(points_list==0); 
    for i = rem_p 
       A = r{1,i};
       if ~isempty(find(already_taken==A(1), 1))
           index = points_list(A(1));
           initial_cluster{index} = [initial_cluster{index},i]; 
           points_list(i) = index;
       elseif  ~isempty(find(already_taken==A(2), 1))
           index = points_list(A(2));
           initial_cluster{index} = [initial_cluster{index},i]; 
           points_list(i) = index;
       end
    end
end
