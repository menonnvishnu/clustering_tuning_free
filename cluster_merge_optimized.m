function z = cluster_merge_optimized(X)
% Inputs:
% X - data matrix n x N containing N data points
% z - intial clustering labels of the datapoints: should be from 1 to L_i
% Output:
% z - output clustering labels from 1 to L_est
X = normc(X); %normalize the columns
z=initial_clustering_final(X); z=z'; %perform initial clustering
L = length(unique(z)); %number of clusters in the initial clustering
theta = abs(acos(X'*X));
%theta(i,j) is the angle between X(:,i) and X(:,j).
%theta is trimmed to upper triangular region (i<=j) so that repetition is avoided

sum_theta=zeros(L); sum_sq_theta=zeros(L); t=zeros(L);
%sum_theta(i,j) - sum of angles formed between data points in  cluster i and cluster j
%sum_sq_theta(i,j) - sum of squares of angles formed between data points in  cluster i and cluster j
%t(i,j)-number of angles formed between data points in  cluster i and cluster j = (N_i+N_j)_C_2
%these three are used to estimate within cluster and between cluster means and variances

%Initialization
for ii=1:L
    theta_k=nonzeros(triu(theta(find(z==ii),find(z==ii)),1)); %unique angles formed between points in cluster ii and cluster jj
    sum_theta(ii,ii)=sum(theta_k); sum_sq_theta(ii,ii)=sum(theta_k.^2); t(ii,ii)=length(theta_k);
    clear theta_k
    for jj=(ii+1):L %to avoid repetitions, jj starts from ii resulting in upper triangular sum_theta, sum_sq_theta and t
        theta_k=nonzeros(theta(find(z==ii),find(z==jj))); %unique angles formed between points in cluster ii and cluster jj
        sum_theta(ii,jj)=sum(theta_k);
        sum_sq_theta(ii,jj)=sum(theta_k.^2);
        t(ii,jj)=length(theta_k);
        clear theta_k
    end
end
sum_theta=(sum_theta+sum_theta').*(ones(L)-1/2*eye(L));
sum_sq_theta=(sum_sq_theta+sum_sq_theta').*(ones(L)-1/2*eye(L));
t=(t+t').*(ones(L)-1/2*eye(L));

dB=B_dist(sum_theta,sum_sq_theta,t); %Bhattacharyya distance
[gamma,ar,ac]=update_params(dB); %minimum Bhattacharya distance and corresponding cluster indices
zeta=1/sqrt(min(t(ar,ar),t(ar,ac))-1); %zeta-threshold on gamma

%Iteratively merge
while gamma<zeta
    % cluster with (larger) label ac is merged to cluster with label ar
    
    temp=sum_theta(ar,ac);%update
    sum_theta(:,ar)=sum_theta(:,ar)+sum_theta(:,ac); %update
    sum_theta(ar,:)=sum_theta(ar,:)+sum_theta(ac,:); %update
    sum_theta(ar,ar)=sum_theta(ar,ar)-temp;%update
    sum_theta(ac,:)=[]; sum_theta(:,ac)=[];%update
    
    temp=sum_sq_theta(ar,ac);%update
    sum_sq_theta(:,ar)=sum_sq_theta(:,ar)+sum_sq_theta(:,ac); %update
    sum_sq_theta(ar,:)=sum_sq_theta(ar,:)+sum_sq_theta(ac,:); %update
    sum_sq_theta(ar,ar)=sum_sq_theta(ar,ar)-temp;%update
    sum_sq_theta(ac,:)=[]; sum_sq_theta(:,ac)=[];%update
    
    temp=t(ar,ac);%update
    t(:,ar)=t(:,ar)+t(:,ac); %update
    t(ar,:)=t(ar,:)+t(ac,:); %update
    t(ar,ar)=t(ar,ar)-temp;%update
    t(ac,:)=[]; t(:,ac)=[];%update

    z(find(z==ac))=ar; z(find(z>ac))=z(find(z>ac))-1; %update clustering labels

    L=L-1;
    dB=B_dist(sum_theta,sum_sq_theta,t);
        
    %find mergable pair for next iteration
    [gamma,ar,ac]=update_params(dB);
    zeta=1/sqrt(min(t(ar,ar),t(ar,ac))-1);
end
%[~,z]=max(z==unique(z)',[],2);

    function dB1 = B_dist(sum_theta1,sum_sq_theta1,t1)
        %function to find Bhattacharyya distance
        %concerned with finding only upper diagonal entries correctly
        %dB1(i,j)=B-distance between 'angles formed by points within cluster i' and 'angles formed between points in cluster i and j'
        m=sum_theta1./t1; %within and between cluster means
        v=(sum_sq_theta1-t1.*(m.^2))./(t1); %within and between cluster variances
        m1=diag(m); v1=diag(v);
        dB1=(((m-m1).^2)./(v+v1)+log((v./v1+v1./v)*0.25+0.5))*0.25; %B-distance between Gaussians
        dB1=dB1+diag(inf*ones(length(t1),1)); 
        %make diagonal and lower diagonal entries to infinity to avoid self distance and repetition  
    end
    function [gamma1,ar2,ac2] = update_params(dB1)
        %find min and argmin of B-distance
        [temp_min,ar1]=min(dB1);
        [gamma1,ac1]=min(temp_min);
        ar1=ar1(ac1);
        ar2=min(ar1,ac1);
        ac2=max(ar1,ac1);
        %(ar1,ac1) is the mergable pair
        %gamma is the B-distance between 'angles formed by points within cluster ar' and 'angles formed between points in cluster ar and ac'
    end
end