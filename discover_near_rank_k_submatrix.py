import random 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict 
import pymbc
import concurrent.futures
import os
from functools import partial
from sklearn.cluster import SpectralBiclustering

class SamplingAlgorithm:
    def __init__(self, D, k,  lambd = 1, tau_u = 2, tau_v = 2,  delta=None, delta_rectangle = None, verbose=False, plot=False, 
                 approximate_biclique = False, sparsity_constraint=False, use_svd = False, increase_every = 10000, increase_factor = 10):
        """
         Find near-rank-k submatrices with approximation guarantees 
        @params:
        D: input array
        k: input target rank 
        lambd: weight size in objective function 
        tau_u: minimum number of rows in extracted maximum-edge biclique (see https://github.com/wonghang/pymbc)
        tau_v: minimum number of columns in extracted maximum-edge biclique (see https://github.com/wonghang/pymbc)
        delta: tolerance parameter (float)
        delta_rectangle: initial tolerance (float)
        verbose: level of verbosity (bool)
        plot: produce plots in addition to standard output (bool)
        approximate_biclique: whether to use an heuristic for the maximum-edge biclique or pymbc (bool)
        sparsity_constraint: whether the input matrix is sparse and we want to avoid solutions of all zeros  (bool)
        use_svd: whether to use the svd to find a rank-k approximation instead of the more interpretable approximation (bool)
        increase_every: number of initial samples before delta_rectangle is increased by increase_factor (int) 
        increase_factor: the multiplicative factor by which delta_rectangle is increased every increase_every iterations (int) 
        """

        self.D = D
        self.k = k 
        self.lambd = lambd 
        self.tau_u = tau_u                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        self.tau_v = tau_v
        self.delta = delta
        self.delta_rectangle = delta_rectangle
        self.n, self.m = D.shape 
        self.row_indices = [j for j in range(D.shape[0])]
        self.verbose = verbose 
        self.plot = plot  
        self.sparsity_constraint = sparsity_constraint   
        self.use_svd = use_svd
        self.increase_factor = increase_factor 
        self.increase_every = increase_every
        self.zero_tolerance = 1e-8 
        

            


    def rank_check(self, matrix): 
        '''
        check whether a matrix is full rank. This function uses singular values rather than determinant, which is equivalent but more convenient. 
        @params: 
        matrix: input (seed) matrix 

        returns: 
        True if the matrix is full rank and False otherwise (bool)
        '''
    
        singular_values = np.linalg.svd(matrix, compute_uv=False)

        num_large_singular_values = np.sum(singular_values > self.delta_rectangle)
        
        # Return True if at least (d+1) singular values are above threshold
        if num_large_singular_values >= self.k + 1: 
            return True, num_large_singular_values
        else: 
            return False, num_large_singular_values


    def sample_points(self, max_attempts=10e10):
        '''
        Sample self.k+1 row indices and column indices in the initialization phase 
        @params: 
        max_attempts = 10e10 (int)

        returns:
        submatrix: self.k + 1 x self.k + 1 initial array (2D array)
        row_indices:  self.k + 1 row indices (1D array)
        col_indices: self.k + 1 col_indices (1D array)
        '''

        for _ in range(max_attempts):
            # Sample self.k+1 unique row and column indices
            row_indices = np.sort(np.random.choice(self.n, self.k + 1, replace=False))
            col_indices = np.sort(np.random.choice(self.m, self.k + 1, replace=False))
            # Extract the submatrix
            submatrix = self.D[np.ix_(row_indices, col_indices)]
            # Finally, check to avoid trivial vectors with all zeros 
            if not ( np.any(np.sum( np.abs(submatrix) < self.zero_tolerance, axis=1) >= self.k-1 ) or np.any(np.sum( np.abs(submatrix) < self.zero_tolerance, axis=0) >= self.k-1 )):
                return submatrix, row_indices, col_indices
            
        raise ValueError("Failed to find a valid submatrix within the allowed attempts.")




    def compute_indicator_matrix(self, this_D, row_indices, column_indices, initial_matrix_rank): 
        '''
        Subroutine to compute the indicator matrix for the rows or the columns in the expansion phase 
        @params: 
        this_D: input matrix  (2D array)
        row_indices: vector of row indices (1D array)
        column_indices: vector of column indices (1D array)
        initial_matrix_rank: rank of the initial matrix  (int)

        return  
        Imat: indicator matrix (2D array)
        linear_combinations_all: matrix of orthogonal projection approximating the input matrix (2D array)
        residuals: matirx of absolute deviations between the input matrix and the orthogonal projections (2D array)
        '''
    
        basis_vectors = random.sample(sorted(row_indices), initial_matrix_rank) 
        #
        A = this_D[np.ix_(basis_vectors, column_indices)].T  

        A_full = this_D[basis_vectors].T 

        A_pseudo_inv = np.linalg.pinv(A)

        coefficients_all = np.dot(this_D[:, column_indices], A_pseudo_inv.T)
        
        if self.plot: 
            plt.imshow(coefficients_all)
            plt.title("Projection Coefficients") 
            plt.show() 
        
        if self.verbose: 
            if coefficients_all.shape != (this_D.shape[0], self.k) : 
                print("shape mismatch ", coefficients_all.shape) 

        linear_combinations_all = np.dot(coefficients_all, A_full.T)

        if self.sparsity_constraint: 
            rows_below_tolerance = np.all(np.abs(linear_combinations_all) < self.zero_tolerance, axis=1)
            linear_combinations_all[rows_below_tolerance] = np.inf

        # Compute residuals 
        residuals = np.abs(this_D - linear_combinations_all) 

        if self.plot:
            plt.imshow(residuals) 
            plt.title("Residuals") 
            plt.show() 

            plt.plot(residuals) 
            plt.title("Residuals") 
            plt.show() 

        # indicator matrix 
        Imat = (residuals < self.delta).astype(bool)

        return Imat , linear_combinations_all, residuals



    def perform_biclustering(self, adj, linear_combinations_all):
        '''
        Perform biclustering to heuristically extract a maximum-edge biclique 
        @params: 
        adj: indicator matrix (2D array)
        linear_combinations_all: orthogonal projections (2D array)
        

        returns: 
        best_approximation: output submatrix approximation (2D array)
        best_indices: output submatrix indices (1D array)
        best_error: output submatrix rank-k approximation error (float)
        best_D_sub: output submatrix (2D array)
        
        '''

        # Apply Spectral Co-Clustering to a re-scaled input indicator matrix 
        model = SpectralBiclustering(n_clusters=2, random_state=42)
        model.fit(   (adj * 100000000 + 0.0000000001)       )

        row_labels = model.row_labels_
        column_labels = model.column_labels_

        best_approximation_loss = float("inf")
        best_approximation = np.array([]) 
        best_error = float("inf")
        best_indices = ([],[]) 

        n_to_iterate_rows = len( set(row_labels) ) 
        n_to_iterate_cols = len( set(column_labels ) ) 

        max_error = 0 
        max_size = 0 

        all_rank_k_approximations = [] 
        all_errors = [] 
        all_sizes = [] 
        all_indices = [] 
        all_D_subs = []

        # Find the best indices 
        for i in range(n_to_iterate_rows): 
            for j in range(n_to_iterate_cols): 

                row_indices = set( np.where(row_labels == i)[0] )
                col_indices = set( np.where(column_labels == j)[0] ) 

                row_indices = list(row_indices) 
                col_indices = list(col_indices)

                D_sub = self.D[np.ix_(row_indices, col_indices)]

                if not self.use_svd: 
                    rank_k_approximation = linear_combinations_all[np.ix_(row_indices, col_indices)] 
                else: 
                    rank_k_approximation = self.rank_d_svd_approximation(D_sub) 
                
                error = ((rank_k_approximation - D_sub)**2).sum() / np.prod(D_sub.shape) 
                
                if error > max_error: 
                    max_error = error 

                this_size = np.prod(rank_k_approximation.shape) 
                if this_size > max_size: 
                    max_size = this_size 

                all_rank_k_approximations.append(rank_k_approximation) 
                all_errors.append(error) 
                all_indices.append( (row_indices, col_indices)  )
                all_sizes.append(this_size) 
                all_D_subs.append(D_sub)

        for h in range(len(all_errors)): 

            loss_value = (all_errors[h] / max_error) - self.lambd * ( all_sizes[h] / max_size )
            
            if loss_value < best_approximation_loss: 
                best_indices = all_indices[h] 
                best_approximation = all_rank_k_approximations[h]
                best_error = all_errors[h] 
                best_D_sub = all_D_subs[h] 
                

        return [best_approximation, best_indices,  best_error, best_D_sub]





    def rank_d_svd_approximation(self, submatrix):
        """
        Computes the rank-d approximation of the given submatrix using SVD.

        parameters:
            submatrix (np.ndarray): The input matrix to approximate.
        returns:
            np.ndarray: The rank-d approximation of the input matrix.
        """
        # Compute full SVD
        U, S, Vt = np.linalg.svd(submatrix, full_matrices=False)
        
        # Keep only the top `d` components
        U_d = U[:, :self.k]   # First d columns
        S_d = np.diag(S[:self.k])  # Convert first d singular values to a diagonal matrix
        Vt_d = Vt[:self.k, :]  # First d rows

        # Compute the rank-d approximation
        A_d = U_d @ S_d @ Vt_d  # Matrix multiplication

        return A_d



    def run(self, extra_argument = None): 
        '''
        Extract a near-rank-k submatrix 
        
        returns: 
        best_approximation: optimal near-rank-k submatrix (2D array)
        best_error: optimal rank-k approximation error for the output submatrix (float) 
        best_indices: output submatrix indices (1D array))
        size: size of the output submatrix (int)
        '''

        ''' initialization phase  '''  
        check = True 
        n_samples = 0 
        while check:              
            submatrix, row_indices, col_indices = self.sample_points()             
            check, initial_matrix_rank = self.rank_check(submatrix)   # Check whether submatrix is full rank 
            n_samples+=1 
            if n_samples % self.increase_every == 0:
                self.delta_rectangle *= self.increase_factor

        if self.verbose: 
            print("found first candidate ")
            print("number of required samples ", n_samples)
            print("row indices ", row_indices)
            print("col indices ", col_indices)
        

        ''' expansion phase  '''  
        # Compute indicator matrix for the rows 
        Imat_rows, linear_combinations_all_rows, residuals_rows = self.compute_indicator_matrix(self.D, row_indices=row_indices, column_indices=col_indices, initial_matrix_rank=initial_matrix_rank)

        # Compute indicator matrix for the columns  
        Imat_columns, linear_combinations_all_columns, residuals_cols = self.compute_indicator_matrix(self.D.T, row_indices=col_indices, column_indices=row_indices, initial_matrix_rank=initial_matrix_rank)

        # Compute indicator matrix (intersection)  
        Imat = (Imat_rows & Imat_columns.T)
       
        if self.plot: 
            plt.imshow(Imat_rows) 
            plt.title("Indicator Matrix Rows")
            plt.show() 

            plt.imshow(Imat_columns.T) 
            plt.title("Indicator Matrix Columns")
            plt.show() 
            
            plt.imshow(Imat) 
            plt.title("Indicator Matrix")
            plt.show() 


        # Pick the best orthogonal projection between rows and columns 
        if np.sum(residuals_rows**2) > np.sum(residuals_cols**2): 
            linear_combinations_all = linear_combinations_all_columns
        else:
            linear_combinations_all = linear_combinations_all_rows  

        # Use spectral biclustering to extract a dense submatrix from the indicator matrix (approximating maximium-edge biclique) 
        if self.approximate_biclique: 
            best_approximation, best_indices,  best_error, best_D_sub = self.perform_biclustering(Imat, linear_combinations_all)

        # Use maximum-edge biclique algorithm of Lyu et al. to extract a submatrix of all ones from the indicator matrix 
        else: 
            try: 
                U = np.array([idx_r for idx_r in range(Imat.shape[0])]).astype(np.uint64) 
                V =  np.array([idx_c for idx_c in range(Imat.shape[1])]).astype(np.uint64)
                if self.verbose: 
                    print("running maximum biclique search...")
                row_indices, col_indices = pymbc.maximum_biclique_search(U,  
                                            V,  
                                            Imat,   
                                            self.tau_u,    # tau_u - these are treshold for size of U and V - very useful 
                                            self.tau_v,    # tau_v
                                            3,    # init_type (STAR)
                                            1,    # init_iter 1 (not used)
                                            np.random.randint(0, 32767), # random seed
                                            True, # use_star
                                            2,    # star_max_iter
                                            3)    # optimize
            
                # Best Approximation
                best_D_sub = self.D[np.ix_(row_indices, col_indices)]
                if not self.use_svd:
                    best_approximation = linear_combinations_all[np.ix_(row_indices, col_indices)]
                else: 
                    best_approximation =  self.rank_d_svd_approximation(best_D_sub)
                best_indices = (row_indices,col_indices)

            except: 
                if self.verbose: 
                    print("maximum biclique search failed. Switching to biclustering.")
                
                # In some cases the C code of pymbc fails - in such cases, we switch to to spectral biclustering 
                best_approximation, best_indices,  best_error, best_D_sub = self.perform_biclustering(Imat, linear_combinations_all)

        best_error = ((best_approximation - best_D_sub)**2).sum() / np.prod(best_D_sub.shape) 

        if self.verbose: 
            print("best_indices ", best_indices) 

        if self.plot: 
            self.plot_approximation(best_approximation, best_D_sub, best_error) 

        return  [best_approximation , best_error , best_indices, np.prod(best_approximation.shape)]
            
    

    def run_multiple(self, N_rep=10): 
        ''' Extract a near-rank-k submatrix for different initializations 
        @params: 
        N_rep: number of initializations to explore  (int)
          
        returns:  
        best_approximation: optimal near-rank-k submatrix (2D array)
        best_error: optimal rank-k approximation error for the output submatrix (float) 
        best_indices: output submatrix indices (1D array)
        best_approximation_loss: optimal near-rank-k submatrix objective  (float) 
        '''



        best_approximation_loss = float("inf") 
        best_approximation = None 
        best_error = None 
        best_indices = (None, None) 

        all_errors = [] 
        all_sizes = [] 
        all_approximations = [] 
        all_indices = [] 
        
        max_error = 0 
        max_size = 0 

        cnt_its = 0 
        while cnt_its < N_rep: 
            rank_k_approximation , error , indices, size = self.run() 
            
            all_approximations.append( rank_k_approximation )
            all_errors.append( error ) 
            all_sizes.append( size ) 
            all_indices.append( indices ) 

            if error > max_error: 
                max_error = error 
            
            if size > max_size:
                max_size = size 

            cnt_its +=1 
           
        for j in range(cnt_its): 
            
            this_error = all_errors[j] 
            this_size = all_sizes[j] 
            loss_value = (this_error / max_error) - self.lambd * (this_size / max_size) 

            if loss_value < best_approximation_loss: 
                best_approximation_loss = loss_value
                best_approximation = all_approximations[j] 
                best_error = all_errors[j]  
                best_indices = all_indices[j] 
        
        return [best_approximation , best_error , best_indices, best_approximation_loss]
        


    def run_multiple_parallel(self, N_rep=10):
        ''' Extract a near-rank-k submatrix for different initializations run in parallel 
        @params: 
        N_rep: number of initializations to explore  (int)
          
        returns:  
        best_approximation: optimal near-rank-k submatrix (2D array)
        best_error: optimal rank-k approximation error for the output submatrix (2D array)
        best_indices: output submatrix indices (1D array)
        best_approximation_loss: optimal near-rank-k submatrix objective  (float) 
        '''
        
        print(f"Available CPU cores: {os.cpu_count()}")

        best_approximation_loss = float("inf")
        best_approximation = None
        best_error = None
        best_indices = (None, None)

        def run_task():
            """Wrapper function to call self.run() and handle exceptions."""
            try:
                return self.run()  # Returns (rank_k_approximation, error, indices, loss_value)
            except:
                print("Error - skip")
                return None

        
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(partial(self.run), range(N_rep)))

        max_error = float("-inf")  
        max_size = float("-inf")   
        
        for element in results:
            if element is not None:
                max_error = max(max_error, element[1])
                max_size = max(max_size, element[3]) 

        for result in results:
            if result:
                rank_k_approximation, error, indices, size = result
                loss_value = error / max_error - self.lambd * (size / max_size) 
                if loss_value < best_approximation_loss:
                    best_approximation_loss = loss_value
                    best_approximation = rank_k_approximation
                    best_error = error
                    best_indices = indices

        return [best_approximation, best_error, best_indices, best_approximation_loss]
    



    def plot_approximation(self, rank_k_approximation, D_sub, error): 
        '''
        Plot approximation and actual submatrix 

        @params
        rank_k_approximation: computed rank-k approximation (array)
        D_sub: submatrix to approximate (array)
        error: approximation error (float) 
        '''
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        #
        # Plot the entire matrix with colorbar
        cax = axs[0].imshow(rank_k_approximation, cmap='viridis', interpolation='none')
        axs[0].set_title("Approximated Values")
        fig.colorbar(cax, ax=axs[0])
        #
        # Plot the entire matrix with colorbar
        cax = axs[1].imshow(D_sub, cmap='viridis', interpolation='none')
        axs[1].set_title("Actual Values")
        fig.colorbar(cax, ax=axs[1])
        #
        plt.title(f"Error: {error:.2f}")
        plt.show()



