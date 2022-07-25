import numpy as np

class clustering():
    
    def __init__(self,X,clustering = [],min_clusters = 3, max_clusters = 50):
        self.config = config
        self.min_no = min_clusters
        self.max_no = max_clusters
        self.A = self.get_angles(X)
        self.num_data = np.shape(self.A)[0]
        self.globalmu = 2*np.sum(self.A)/(self.num_data*(self.num_data-1))
        self.A = self.A + 10*np.eye(self.num_data)
        self.D = np.zeros(1)
        self.gammas= []
        self.zetas = []
        if len(clustering) == 0:
            self.clustering = self.init_cluster()
        else:
            self.clustering = clustering
        
    def get_angles(self,X):
        X = X/np.linalg.norm(X,axis=0)
        A = np.matmul(X.T,X)
        A = np.arccos(np.clip(A, -1, 1))
        
        return A
    
    def init_cluster(self):
        indices_left = list(np.arange(self.num_data))
        clusters = []
        k = 0
        L = self.num_data
        while L>=3:
            if k in indices_left:
                angles = self.A[k,:]
                nbrs = np.argsort(angles)
                nbrs = [n for n in nbrs if n in indices_left]
                nbrs = np.asarray(nbrs)[0:2]
                indices = np.append(nbrs,k)
                clusters.append(indices)
                indices_left = [index for index in indices_left if index not in indices]
                L-=3
            k+=1
        return clusters
            
         
    def estimate_mv(self,c1_indices,c2_indices):
        
        A_sub = self.A[c1_indices][:,c2_indices]
        A_sub = np.reshape(A_sub,(np.size(A_sub),1))
        A_sub = A_sub[A_sub!=10]
        
        lb = len(A_sub)
        
        mu= np.mean(A_sub)
        v = np.var(A_sub)*(lb/(lb-1))    
        
        return mu,v
    
    def get_score(self,within_params,between_params):
        
        [mub,vab] = within_params
        [muw,vaw] = between_params
        score =  max(0,0.25*(  ((mub-muw)**2/(vab+vaw+1e-8)) + np.log( 0.25*((vab/(vaw+1e-8)) + (vaw/(vab+1e-8))) + 0.5 )  ))
        
        return score
    
    def get_init_D(self):
        
        L = len(self.clustering)
        D = np.inf*np.ones((L,L))
        params_within = []
        mu_between = np.zeros((L,L))
        var_between = np.zeros((L,L))
            
        for i in range(L):
            mu,v = self.estimate_mv(self.clustering[i],self.clustering[i])
            params_within.append([mu,v])
            for j in range(L):
                if i!=j:
                    mu,v = self.estimate_mv(self.clustering[i],self.clustering[j])
                    mu_between[i,j] = mu
                    var_between[i,j] = v
                    D[i,j] = self.get_score(params_within[i],[mu,v])
                    
        return D
            
    def update_D(self,index0,index1):
        
        mu,v = self.estimate_mv(self.clustering[index0],self.clustering[index0])
        params_within = [mu,v]
        for i in range(np.shape(self.D)[0]):
            if i != index0 and i != index1:
                mu1,v1 = self.estimate_mv(self.clustering[index0],self.clustering[i])
                self.D[index0,i] = self.get_score(params_within,[mu1,v1])
        
        for i in range(np.shape(self.D)[0]):
            if i != index0 and i != index1:
                mu1,v1 = self.estimate_mv(self.clustering[i],self.clustering[index0])
                self.D[i,index0] = self.get_score(params_within,[mu1,v1])        
        
        self.D = np.delete(self.D, index1, 0)
        self.D = np.delete(self.D, index1, 1)

            
    def merge_clustering(self):
        s_min = np.min(self.D)
        mergable = np.unravel_index(self.D.argmin(), self.D.shape)
        c_new = np.append(self.clustering[mergable[0]],self.clustering[mergable[1]])
        thr = 1/np.sqrt((min(len(self.clustering[mergable[0]]),len(self.clustering[mergable[1]])))-1)
        
        if mergable[0] < mergable[1]:
            self.clustering[mergable[0]] = c_new
            self.update_D(mergable[0],mergable[1])
            del self.clustering[mergable[1]]
        else:
            self.clustering[mergable[1]] = c_new
            self.update_D(mergable[1],mergable[0])
            del self.clustering[mergable[0]]
        
        return s_min,thr
    
    def fit(self):
        self.D = self.get_init_D()
        
        s_min = 0
        thr = 1
        self.gammas= []
        self.zetas = []
        while (s_min<=thr and len(self.clustering)>self.min_no) or len(self.clustering)>self.max_no:
            s_min,thr = self.merge_clustering()
            self.gammas.append(s_min)
            self.zetas.append(thr)
            
        return self.clustering
