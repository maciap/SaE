import random 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict 
import pymbc
import time 
import concurrent.futures
import os
from functools import partial
from sklearn.cluster import SpectralBiclustering
import networkx as nx 

class SamplingAlgorithm:
    def __init__(self, D,  lambd = 1, tau_u = 2, tau_v = 2, delta=None, delta_rectangle = None, verbose=False, plot=False, 
                 approximate_biclique=False, sparsity_constraint=False, use_svd = True,  increase_every = 10000, increase_factor = 10):
        """
        Find near-rank-1 submatrices with approximation guarantees 
        @params:
        D: input array
        lambd: weight size in objective function 
        tau_u: minimum number of rows in extracted maximum-edge biclique (see https://github.com/wonghang/pymbc)
        tau_v: minimum number of columns in extracted maximum-edge biclique (see https://github.com/wonghang/pymbc)
        delta: tolerance parameter (float)
        delta_rectangle: initial tolerance (float)
        verbose: level of verbosity (bool)
        plot: produce plots in addition to standard output (bool)
        approximate_biclique: whether to use an heuristic for the maximum-edge biclique or pymbc (bool)
        sparsity_constraint: whether the input matrix is sparse and we want to avoid solutions of all zeros  (bool)
        use_svd: whether to use the svd to find a rank-1 approximation instead of the more interpretable approximation (bool)
        increase_every: number of initial samples before delta_rectangle is increased by increase_factor (int) 
        increase_factor: the multiplicative factor by which delta_rectangle is increased every increase_every iterations (int) 
        """
        self.D = D
        self.verbose = verbose 
        self.approximate_biclique = approximate_biclique  #Use Lyu et al. , "Maximum Biclique Search at Billion Scale" to mine maximum bicliques if False otherwise it performs area sampling 
        self.plot = plot 
        self.delta = delta # threshold for ratios  
        self.delta_rectangle = delta_rectangle # threshold for initial determinant "
        self.lambd = lambd 
        self.tau_u = tau_u   # nimimum number of rows for "Maximum Biclique Search at Billion Scale"  (only used if approximate_biclique is False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        self.tau_v = tau_v  # minimum number of columns for "Maximum Biclique Search at Billion Scale"  (only used if approximate_biclique is False) 
        self.sparsity_constraint = sparsity_constraint 
        self.total_entries = np.prod(self.D.shape) 
        self.increase_every = increase_every 
        self.increase_factor = increase_factor
        self.use_svd = use_svd

     

    def sample_first_rectangle(self): 
        '''Sample initial 2x2 matrices
        
        returns: 
        top_left, bottom_left, top_right, bottom_right: sampled entries
        '''
        top_left, bottom_right = self.sample_first_pairs() 
        top_left, bottom_left, top_right, bottom_right = self.complete_rectangle(top_left, bottom_right) 
        return top_left, bottom_left, top_right, bottom_right


    def sample_first_pairs(self):
        """
        Sample two distinct pairs of indices (row, col) from the matrix self.D.
        
        returns: 
        (i1, j1), (i2, j2): two tuples of indices in the form ((i1, j1), (i2, j2))
        """
        if len(self.D) == 0 or len(self.D[0]) == 0:
            raise ValueError("Matrix D is empty or not properly initialized.")
        num_rows = len(self.D)
        num_cols = len(self.D[0])
        
        # Sample the first pair of indices
        i1, j1 = random.randint(0, num_rows - 1), random.randint(0, num_cols - 1)
        
        # Sample the second pair ensuring it is different from the first pair
        while True:
            i2, j2 = random.randint(0, num_rows - 1), random.randint(0, num_cols - 1)
            #if (i2, j2) != (i1, j1):
            if i2 != i1 and j2 != j1: 
                break
            
        return (i1, j1), (i2, j2) 


    def complete_rectangle(self, top_left, bottom_right): 
        '''
        Given two entries, complete a 2x2 initial submatrix 
        @params: 
        top_left : sampled index of initial matrix entry (tuple)
        bottom_right : sampled index of initial matrix entry (tuple)


        return 
        top_left, bottom_left, top_right, bottom_right: the indices of the 2x2 submatrix 
        '''
        i1, j1 = top_left
        i2, j2 = bottom_right
        top_right = (i1, j2)
        bottom_left = (i2, j1) 
        return (top_left, bottom_left, top_right, bottom_right) 
    

    def check_rectangle(self, top_left, bottom_left, top_right, bottom_right):
        '''
        Check the initial determinant condition 
        @params: 
        top_left: index of initial matrix entry (tuple)
        bottom_left: index of initial matrix entry (tuple)
        top_right: index of initial matrix entry (tuple)
        bottom_right: index of initial matrix entry (tuple)



        returns: 
        boolean indicating whether the initial determinant condition is satisfied 
        '''

        if (abs(self.D[top_left] ) < 1e-4 or abs(self.D[bottom_right]) < 1e-4 or  abs(self.D[top_right] ) < 1e-4 or abs(self.D[bottom_left]) < 1e-4):
             return False
        elif self.delta_rectangle  is None: 
            return self.D[top_left] * self.D[bottom_right] == self.D[top_right] * self.D[bottom_left]
        else: 
            return np.abs( self.D[top_left] * self.D[bottom_right] - self.D[top_right] * self.D[bottom_left] ) <= self.delta_rectangle

          
    def compute_ratios(self, row_idx):
        '''
        Compute row-wise ratios 
        @params: 
        row_idx: index of anchor row 

        returns: 
        matrix of row-wise ratios
        '''
        denominator = self.D[row_idx].copy()
        denominator[denominator == 0] += 1e-5 
        return self.D / denominator

    def compute_ratios_col(self, col_idx): 
        '''
        Compute column-wise ratios 
        @params: 
        col_idx: index of anchor column 

        returns: 
        matrix of column-wise ratios
        '''
        denominator = self.D[:, col_idx][:, np.newaxis].copy()
        denominator[denominator == 0] += 1e-5 
        return  self.D  / denominator


    def process_rows_and_column_ratios(self, R, sampled_index, necessary_index, rows=True):
        '''
        Find subsets of near-constant ratios in ratio matrices 
        @params: 
        R: matrix of ratios (array) 
        sampled_index: index of the anchor row (if rows=True) or column (if rows=False)
        necessary_index: index of the anchor row (if rows=False) or column (if rows=True)
        rows: boolean indicating whether to compute row-wise or column-wise ratios 


        returns:
        sets of indices of near-constant ratios in each row  
        '''

        result = [set([]) for _ in range(R.shape[0])] # emptyset   
        if rows: 
            for idx_row, row in enumerate(R): 
                if idx_row!=sampled_index:
                    subset = self.find_best_subset_with_necessary_element(row, self.delta, necessary_index) 
                    if self.sparsity_constraint: # avoid all 0s in sparse matrices 
                        if np.all(np.isclose([self.D[idx_row][cnt_col] for cnt_col in subset], 0, atol=1e-6)): 
                            subset = []    

                    result[idx_row] = set(subset)
                
        else: 
            for idx_col, col in enumerate(R.T):  # Transpose the matrix to get columns as rows
                if idx_col!=sampled_index:
                    subset = self.find_best_subset_with_necessary_element(col, self.delta, necessary_index) 
                    if self.sparsity_constraint: # avoid all 0s in sparse matrices 
                        if np.all(np.isclose([self.D[:,idx_col][cnt_row] for cnt_row in subset], 0, atol=1e-6)): 
                            subset = []    

                    for el in subset: 
                        result[el].add(idx_col)

        return result 
    


    def find_best_subset_with_necessary_element(self, vec, delta, necessary_index):
        """
        Given a vector, find the best non-contiguous subset where:
        - The absolute difference between the max and min values is less than or equal to delta.
        - A necessary element at index `necessary_index` must be included in the subset.

        @params:
        - vec: 1D row or column vector (array)
        - delta: tolerance, the maximum allowed absolute difference between max and min in the subset (float).
        - necessary_index: int, the index of the element that must be included in the subset, i.e., index of the anchor row (if rows=False) or column (if rows=True).

        returns:
        subset_indices: list of indices for the best non-contiguous subset.
        """

        necessary_element = vec[necessary_index]
        subset_indices = [necessary_index]
        max_element = necessary_element
        min_element = necessary_element

        # Sort the row based on proximity to the necessary element
        # We need to keep track of both value and original index, hence use enumerate
        sorted_row_with_indices = sorted(enumerate(vec), key=lambda x: abs(x[1] - necessary_element))

        # Add elements to the subset while the condition on delta holds
        for index, element in sorted_row_with_indices:
            if index == necessary_index:
                continue  # Skip the necessary element since it's already included
            
            # Calculate new min and max based on absolute values if this element is added
            new_max = max(max_element, element)
            new_min = min(min_element, element)

            # Check if the absolute difference between max and min exceeds delta
            if abs(new_max - new_min) <= delta:
                subset_indices.append(index)
                max_element = new_max
                min_element = new_min
            else:
                break  # Stop when the range exceeds delta
        return subset_indices



    def compute_rank_one_approximation(self, row_indices, col_indices): 
        '''
        Compute an intepretable rank-1 approximation based on either the anchor row or the anchor column (the approximation for which we derive approximation guarantees) or the rank-one svd 

        @params: 
        row_indices: output submatrix row indices 
        col_indices:  output submatrix column indices

        returns: 
        rank_one_approximation: computed rank-one approximation (array)
        error: approximation error (float) 
        D_sub: submatrix to approximate (array)
        '''

        D_sub = self.D[np.ix_(row_indices, col_indices)]
        if not self.use_svd: 
            # Row-based rank-1 approximation 
            last_row_idx = row_indices[-1]
            reference_row = self.D[last_row_idx, col_indices] 
            # Project all rows onto reference_row
            row_projections = self.D[np.ix_(row_indices, col_indices)] @ reference_row.T
            ref_row_norm_sq = np.dot(reference_row, reference_row)
            if ref_row_norm_sq < 1e-8:
                row_based_approx = np.zeros_like(D_sub)
                row_error = np.inf
            else:
                coeffs = row_projections / ref_row_norm_sq  
                row_based_approx = np.outer(coeffs, reference_row)
                row_error = np.mean((D_sub - row_based_approx) ** 2)

            # Column-based rank-1 approximation 
            last_col_idx = col_indices[-1]
            reference_col = self.D[row_indices, last_col_idx]  
            col_projections = self.D[np.ix_(row_indices, col_indices)].T @ reference_col
            ref_col_norm_sq = np.dot(reference_col, reference_col)
            if ref_col_norm_sq < 1e-8:
                col_based_approx = np.zeros_like(D_sub)
                col_error = np.inf
            else:
                coeffs = col_projections / ref_col_norm_sq 
                col_based_approx = np.outer(reference_col, coeffs)
                col_error = np.mean((D_sub - col_based_approx) ** 2)

            # Choose the one with lower error
            if row_error <= col_error:
                return row_based_approx, row_error, D_sub
            else:
                return col_based_approx, col_error, D_sub
                       

        else: 
            # Use SVD  
            rank_one_approximation = self.rank_one_svd_approximation(D_sub) 
            # Error 
            error = ((rank_one_approximation - D_sub)**2).sum() / np.prod(D_sub.shape) 

        return rank_one_approximation, error, D_sub
    




    
    def compute_rank_one_approximation_v2(self, row_indices, col_indices): 
        '''
        Compute an intepretable rank-1 approximation as the outer product of an actual row and column or the rank-1 SVD 

        @params: 
        row_indices: output submatrix row indices 
        col_indices:  output submatrix column indices

        returns: 
        rank_one_approximation: computed rank-one approximation (array)
        error: approximation error (float) 
        D_sub: submatrix to approximate (array)
        '''

        if not self.use_svd: 
            # We can choose any row and column, for simplicity we choose the first. 
            one_row_approximated = self.D[row_indices[0],:] 
            one_column_approximated = self.D[:,col_indices[0]] 

            # Ensure that the vectors have non-zero norm 
            i=1
            tmp = one_row_approximated[col_indices]
            while np.linalg.norm(tmp) < 1e-8: 
                one_row_approximated = self.D[row_indices[i],:] 
                tmp = one_row_approximated[col_indices] 
                i+=1
            
            j=1 
            tmp = one_column_approximated[row_indices]
            while np.linalg.norm(tmp) < 1e-8: 
                one_column_approximated = self.D[:,col_indices[j]] 
                tmp = one_column_approximated[row_indices]
                j+=1


            rank_one_approximation = np.outer(one_column_approximated[row_indices], one_row_approximated[col_indices])
            
            # Extract submatrix  
            D_sub = self.D[np.ix_(row_indices, col_indices)]

        
            # Rescale rank-1 approximation  
            optimal_scaling = self.compute_optimal_alpha(D_sub, one_column_approximated[row_indices], one_row_approximated[col_indices])
            rank_one_approximation = optimal_scaling * rank_one_approximation 
            # Compute error  
            error = ((rank_one_approximation - D_sub)**2).sum() / np.prod(D_sub.shape) 

        else: 
            D_sub = self.D[np.ix_(row_indices, col_indices)]

            # Use SVD  
            rank_one_approximation = self.rank_one_svd_approximation(D_sub) 
            # Error 
            error = ((rank_one_approximation - D_sub)**2).sum() / np.prod(D_sub.shape) 

        return rank_one_approximation, error, D_sub
    


    
    
      
    def plot_first_rectangle(self, highlight_indices, highlight_color='red', default_color='blue'):
        """
        Plot a matrix with specified indices highlighted.

        Parameters:
        - rows (int): Number of rows in the matrix.
        - cols (int): Number of columns in the matrix.
        - highlight_indices (list of tuples): List of (row, col) pairs to highlight.
        - highlight_color (str): Color for highlighted cells.
        - default_color (str): Color for default cells.
        """

        rows, cols = self.D.shape 
        # Create a matrix filled with 0 for default entries
        matrix = np.zeros((rows, cols))

        # Set highlighted cells to 1
        for (i, j) in highlight_indices:
            matrix[i, j] = 1

        # Define a color map with only two colors: one for 0 and one for 1
        cmap = matplotlib.colors.ListedColormap([default_color, highlight_color])

        # Plotting
        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap=cmap, interpolation='none')
        
        # Add grid for clarity
        plt.grid(color='black', linestyle='-', linewidth=0.5)

        # Set ticks for clarity
        plt.xticks(np.arange(cols))
        plt.yticks(np.arange(rows))
        
        # Title and axis labels
        plt.title("Matrix with Highlighted Rectangle")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()




    def compute_optimal_alpha(self, D_s, u, v):
        """
        Computes the optimal alpha that minimizes the Frobenius norm of (D_s - alpha * u * v^T).

        @params:
        D_s: 2D numpy array representing the matrix D_s.
        u: 1D numpy array representing the vector u (column from D_s).
        v: 1D numpy array representing the vector v (row from D_s).

        returns:
        alpha: optimal scalar alpha (float).
        """
        # Ensure u and v are column vectors
        u = u.reshape(-1, 1)  # Convert to column vector
        v = v.reshape(-1, 1)  # Convert to column vector


        # Compute numerator: u^T * D_s * v
        numerator = u.T @ D_s @ v

        # Compute denominator: ||u||^2 * ||v||^2
        u_norm_squared = np.linalg.norm(u) ** 2
        v_norm_squared = np.linalg.norm(v) ** 2

        denominator = u_norm_squared * v_norm_squared

        # Compute alpha
        alpha = (numerator / denominator).item()  

        return alpha

    def plot_approximation(self, rank_one_approximation, D_sub, error): 
        '''
        Plot approximation and actual submatrix 

        @params
        rank_one_approximation: computed rank-one approximation (array)
        D_sub: submatrix to approximate (array)
        error: approximation error (float) 
        '''

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        #
        # Plot the entire matrix with colorbar
        cax = axs[0].imshow(rank_one_approximation, cmap='viridis', interpolation='none')
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
		

    def rank_one_svd_approximation(self, submatrix):
        """
        Computes the rank-1 approximation of the given submatrix using SVD.

        @params:
        submatrix: the input matrix to approximate (array)


        returns:
        A_d: the rank-1 approximation of the input matrix (array)
        """
        # Compute full SVD
        U, S, Vt = np.linalg.svd(submatrix, full_matrices=False)
        
        U_d = U[:, 0]  # First column as a column vector
        S_d = S[0]  # First singular value (scalar)
        Vt_d = Vt[0, :]#  # First row as a row vector

        # Compute the rank-1 approximation
        A_d = S_d * np.outer(U_d, Vt_d)  # Equivalent to U_d @ np.diag([S_d]) @ Vt_d

        return A_d



    def perform_biclustering(self, adj):
        '''
        Perform biclustering to heuristically extract a maximum-edge biclique 
        @params: 
        adj: indicator matrix 
        

        returns: 
        best_approximation: output submatrix approximation 
        best_indices: output submatrix indices 
        best_error: output submatrix rank-1 approximation error 
        best_D_sub: output submatrix 
        '''

        
        # Apply Spectral Co-Clustering
        model = SpectralBiclustering(n_clusters=2, random_state=42, method='log', n_components=2, n_best=2)
        model.fit(   (adj * 100000000 + 0.0000000001)       ) 

        row_labels = model.row_labels_
        column_labels = model.column_labels_

        best_approximation_loss = float("-inf")
        best_approximation = np.array([]) 
        best_error = float("inf")
        best_indices = ([],[]) 
        best_D_sub = None 

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
                
                rank_one_approximation, lr_score, D_sub = self.compute_rank_one_approximation(row_indices, col_indices)
        
                this_size = np.prod(rank_one_approximation.shape) 
               
                all_rank_k_approximations.append(rank_one_approximation) 
                all_errors.append(lr_score) 
                all_indices.append( (row_indices, col_indices)  )
                all_sizes.append(this_size) 
                all_D_subs.append(D_sub)


        avg_error = np.mean(all_errors) 
        std_error = np.std(all_errors)
        normalized_errors = (all_errors - avg_error) / std_error 
        
        avg_size = np.mean(all_sizes) 
        std_size = np.std(all_sizes)
        normalized_sizes =  (all_sizes - avg_size) / std_size 

        for h in range(len(all_rank_k_approximations)): 

            loss_value = min( normalized_errors[h], normalized_sizes[h] )
            if loss_value > best_approximation_loss: 
                best_indices = all_indices[h] 
                best_approximation = all_rank_k_approximations[h]
                best_error = all_errors[h] 
                best_D_sub = all_D_subs[h] 
                

        return [best_approximation, best_indices,  best_error, best_D_sub]



    def project_bipartite(self, adj_matrix, project_on="left"):
        """
        Creates a one-mode projection of a bipartite graph from its adjacency matrix.

        @params:
        adj_matrix (numpy.ndarray): Bipartite adjacency matrix (left_nodes x right_nodes)
        project_on (str): "left" to project onto the left set, "right" for the right set.

        returns:
        projected graph with isolated nodes included
        """
        left_nodes, right_nodes = adj_matrix.shape

        if project_on == "left":
            nodes_to_project = left_nodes
            projection = adj_matrix @ adj_matrix.T  # A * A^T projects on left set
        else:
            nodes_to_project = right_nodes
            projection = adj_matrix.T @ adj_matrix  # A^T * A projects on right set

        # Create graph from adjacency matrix
        G = nx.Graph()

        # Add all nodes (including isolated ones)
        G.add_nodes_from(range(nodes_to_project))

        # Add edges where projection matrix is nonzero
        for i in range(nodes_to_project):
            for j in range(i + 1, nodes_to_project):  # Avoid duplicates
                if projection[i, j] > 0:
                    G.add_edge(i, j, weight=projection[i, j])

        return G
    

    def _greedy_plus_plus(self, G, iterations):
        '''
        Extract denseset subgraphs from input graph 
        @params: 
        G: input graph 
        iterations: number of iterations to be executed 


        returns: 
        best_density: densest subgraph density 
        best_subgraph: node in denseset subgraph 
        '''


        if G.number_of_edges() == 0:
            return 0.0, set()
        if iterations < 1:
            raise ValueError(
                f"The number of iterations must be an integer >= 1. Provided: {iterations}"
            )

        loads = {node: 0 for node in G.nodes}  # Load vector for Greedy++.
        best_density = 0.0  # Highest density encountered.
        best_subgraph = set()  # Nodes of the best subgraph found.

        for _ in range(iterations):
            # Initialize heap for fast access to minimum weighted degree.
            heap = nx.utils.BinaryHeap()

            # Compute initial weighted degrees and add nodes to the heap.
            for node, degree in G.degree:
                heap.insert(node, loads[node] + degree)

            # Set up tracking for current graph state.
            remaining_nodes = set(G.nodes)
            num_edges = G.number_of_edges()
            current_degrees = dict(G.degree)

            while remaining_nodes:
                num_nodes = len(remaining_nodes)

                # Current density of the (implicit) graph
                current_density = num_edges / num_nodes

                # Update the best density.
                if current_density > best_density:
                    best_density = current_density
                    best_subgraph = set(remaining_nodes)

                # Pop the node with the smallest weighted degree.
                node, _ = heap.pop()
                if node not in remaining_nodes:
                    continue  # Skip nodes already removed.

                # Update the load of the popped node.
                loads[node] += current_degrees[node]

                # Update neighbors' degrees and the heap.
                for neighbor in G.neighbors(node):
                    if neighbor in remaining_nodes:
                        current_degrees[neighbor] -= 1
                        num_edges -= 1
                        heap.insert(neighbor, loads[neighbor] + current_degrees[neighbor])

                # Remove the node from the remaining nodes.
                remaining_nodes.remove(node)

        return best_density, set(best_subgraph) 


    def project_and_find_densest_subgraph(self, adj_matrix):
        """
        Computes the one-mode projection of a bipartite graph and extracts the densest subgraph using NetworkX.

        @param
        adj_matrix: a NumPy array or SciPy sparse matrix representing the adjacency matrix of a bipartite graph
        
        
        return:
        densest subgraph of the projected graph
        """
        rows, cols = adj_matrix.shape

        # Compute weighted projection graph
        projected_graph_left = self.project_bipartite(adj_matrix, "left")
        
        # Extract the densest subgraph using NetworkX built-in function
        _, densest_subgraph_left = self._greedy_plus_plus(projected_graph_left, iterations = 10)

        projected_graph_right = self.project_bipartite(adj_matrix, "right")

        _, densest_subgraph_right = self._greedy_plus_plus(projected_graph_right, iterations = 10)

        return list(densest_subgraph_left), list(densest_subgraph_right), projected_graph_right




    def greedy_simple_heuristic(self, Imat): 
        '''
        Simple greedy iterative appraoch to extract dense subgraphs
        @params 
        Imat: indicator matrix 


        Returns: 
        best_indices: indices of the output submatrix 
        '''


        all_lowrankness = [] 
        all_size = [] 
        all_indices = [] 
        max_lowrankness = float("-inf") 
        max_size = float("-inf") 
        best_score = float("-inf") 

        for density_theshold in [0.9, 0.95, 0.975]: 
            row_indices, col_indices = self.find_largest_submatrix_density_threshold(Imat, density_threshold=density_theshold)

            if len(row_indices)>0 and len(col_indices)>0: 
                singular_values = np.linalg.svd(self.D[np.ix_(row_indices, col_indices)], compute_uv=False)                    
                # Compute the ratio 
                low_rankeness  = (singular_values[0] ** 2) / np.sum(singular_values ** 2)
                all_lowrankness.append(low_rankeness)
                size =  len(row_indices) * len(col_indices)  #np.prod(final_submatrix.shape) 
                all_size.append(size) 
                all_indices.append((row_indices, col_indices)) 

                if low_rankeness > max_lowrankness:
                    max_lowrankness = low_rankeness
                if size > max_size: 
                    max_size = size 

        avg_error = np.mean(all_lowrankness) 
        std_error = np.std(all_lowrankness)
        normalized_errors = (all_lowrankness - avg_error) / std_error 
        
        avg_size = np.mean(all_size) 
        std_size = np.std(all_size)
        normalized_sizes =  (all_size - avg_size) / std_size 
        
        # Find the best 
        for i in range(len(all_indices)): 

            low_rankeness = normalized_errors[i] 
            this_size = normalized_sizes[i] 
            score =  min(low_rankeness, this_size)   #(low_rankeness / max_lowrankness)  #+  (size / max_size) 

            if score > best_score: 
                best_score = score 
                best_indices = all_indices[i] 
                best_score = score 
                size_of_best = this_size 
                lowrankness_of_best = low_rankeness

        return best_indices


            
            



    def find_largest_submatrix_density_threshold(self, Imat, max_iterations=10000000000, density_threshold=0.3):
        """Find the largest approximate submatrix of ones using greedy pruning with dynamic sum updates.
        
        @params:
        X: binary matrix (numpy array) where 1 represents presence and 0 absence
        max_iterations: maximum number of iterations before stopping
        density_threshold: the density limit for stopping (e.g., 0.2 for 20%)
        
        returns:
        rows: indices of selected rows of output submatrix
        cols: indices of selected columns of output submatrix
        """
        
        X = Imat.copy()  # Avoid modifying the original matrix
        rows, cols = set(range(X.shape[0])), set(range(X.shape[1]))

        # Compute initial row and column sums
        row_sums = np.sum(X, axis=1)
        col_sums = np.sum(X, axis=0)

        # Compute total ones initially
        total_ones = np.sum(X)
        iteration = 0  # Track iterations

        while iteration < max_iterations:
            if not rows or not cols:
                break  # No more rows/cols left

            # Find the row/column with the minimum number of ones
            min_row = min(rows, key=lambda r: row_sums[r], default=None)
            min_col = min(cols, key=lambda c: col_sums[c], default=None)

            # Compute the current submatrix density
            num_elements = len(rows) * len(cols)
            if num_elements == 0:
                break  # Avoid division by zero

            density = total_ones / num_elements  # Compute density

            # STOP CONDITION: If the density rises above the threshold
            if density > density_threshold:
                break  # Stop early

            # Remove the sparsest row or column (whichever has fewer ones)
            if row_sums[min_row] <= col_sums[min_col]:
                # Remove row
                rows.remove(min_row)
                for c in cols:  # Update column sums dynamically
                    col_sums[c] -= X[min_row, c]
                total_ones -= row_sums[min_row]  # Update total ones count
            else:
                # Remove column
                cols.remove(min_col)
                for r in rows:  # Update row sums dynamically
                    row_sums[r] -= X[r, min_col]
                total_ones -= col_sums[min_col]  # Update total ones count

            iteration += 1  # Increment iteration count

        return list(rows), list(cols)#, final_submatrix, density





    
    def run(self, extra_argument = None): 
        '''
        Extract a near-rank-1 submatrix 
        
        returns: 
        best_approximation: optimal near-rank-1 submatrix 
        best_error: optimal rank-1 approximation error for the output submatrxi 
        best_indices: output submatrix indices 
        size: size of the output submatrix 
        '''

        # Initialization 
        check = False 
        n_samples = 0 
        while not check: 
            top_left, bottom_left, top_right, bottom_right = self.sample_first_rectangle() 
            check = self.check_rectangle(top_left, bottom_left, top_right, bottom_right)

            n_samples+=1 
            if n_samples % self.increase_every == 0:
                self.delta_rectangle *= self.increase_factor
                if self.verbose: 
                    print("increasing initial delta ", self.delta_rectangle)


        if self.verbose: 
            print("found initial matrix ")
            print("number of required samples ", n_samples)
            print("top left ", top_left) 
            print("top right ", top_right) 
            print("bottom left ", bottom_left) 
            print("bottom right ", bottom_right) 

         
        # Compute ratios by row 
        ratios_by_row = self.compute_ratios(top_left[0]) 
      
        # Compute ratios by columns 
        ratios_by_columns = self.compute_ratios_col(top_left[1]) 
      
        satisfying_subsets_rows = self.process_rows_and_column_ratios(ratios_by_row, top_left[0], top_left[1], rows=True)
        satisfying_subsets_cols = self.process_rows_and_column_ratios(ratios_by_columns, top_left[1], top_left[0], rows=False)
        intersection_of_indices = [s1.intersection(s2) for s1, s2 in zip(satisfying_subsets_rows, satisfying_subsets_cols)] # the results are given as a list 

        # Compute indicator matrix 
        Imat = np.zeros_like((self.D), dtype="bool") 
        for row_index in range(self.D.shape[0]): 
            inters = satisfying_subsets_rows[row_index].intersection(satisfying_subsets_cols[row_index]) # this are the common eleemnts 
            for elem in inters: 
                Imat[row_index,elem] = 1 
    
        if self.approximate_biclique=="proj": # Use projected denseset subgraph 
            row_indices, col_indices, _  = self.project_and_find_densest_subgraph(Imat)
            best_indices = (row_indices, col_indices) 
            best_approximation, best_error, best_D_sub = self.compute_rank_one_approximation(row_indices, col_indices)

        elif self.approximate_biclique == "greedy": # Use greedy heuristic  
            row_indices, col_indices = self.greedy_simple_heuristic(Imat) 
            best_indices = (row_indices, col_indices) 
            best_approximation, best_error, best_D_sub = self.compute_rank_one_approximation(row_indices, col_indices)


        elif self.approximate_biclique == "biclustering": # Use biclustering 
            best_approximation, best_indices,  best_error, best_D_sub = self.perform_biclustering(Imat) 

        
        else: # Use pymbc 
            try: 
                U = np.array([idx_r for idx_r in range(Imat.shape[0])]).astype(np.uint64) 
                V =  np.array([idx_c for idx_c in range(Imat.shape[1])]).astype(np.uint64)

                if self.verbose: 
                    print("running maximum biclique search!")
                
                row_indices, col_indices = pymbc.maximum_biclique_search(U,  
                                            V,    # V
                                            Imat,    # D
                                            self.tau_u,    # tau_u - these are treshold for size of U and V - very useful 
                                            self.tau_v,    # tau_v
                                            0,    # init_type (STAR)
                                            1,    # init_iter 1 (not used)
                                            np.random.randint(0, 32767), # random seed
                                            True, # use_star
                                            2,    # star_max_iter
                                            3)    # optimize        

                row_indices.append(top_left[0])
                col_indices.append(top_left[1])
                best_indices = (row_indices,col_indices)
                best_approximation, best_error, best_D_sub = self.compute_rank_one_approximation(row_indices, col_indices)
            
            
            except: 
                if self.verbose: 
                    print("maximum biclique search failed. Switching to biclustering.")
                best_approximation, best_indices,  best_error, best_D_sub = self.perform_biclustering(Imat) 

        if self.verbose: 
            print("best_indices ", best_indices) 

        if self.plot: 
            self.plot_approximation(best_approximation, best_D_sub, best_error) 

        return  [best_approximation , best_error , best_indices,  np.prod(best_approximation.shape)]
    


    def run_multiple(self, N_rep=10): 
        ''' Extract a near-rank-1 submatrix for different initializations 
        @params: 
        N_rep: number of initializations to explore  
          
        returns:  
        best_approximation: optimal near-rank-1 submatrix 
        best_error: optimal rank-1 approximation error for the output submatrxi 
        best_indices: output submatrix indices 
        best_approximation_loss: optimal near-rank-1 submatrix objective  
        '''

        best_approximation_loss = float("inf") 
        best_approximation = None 
        best_error = None 
        best_indices = (None, None) 

        all_errors = [] 
        all_sizes = [] 
        all_approximations = [] 
        all_indices = [] 
        all_imats = []
        
        max_error = 0 
        max_size = 0 

        cnt_its = 0 
        while cnt_its < N_rep: 
            rank_one_approximation , error , indices, size = self.run() 
            all_approximations.append( rank_one_approximation )
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
        ''' Run for multiple initializations in parallel
         @params: 
        N_rep: number of initializations to explore  
          
        returns:  
        best_approximation: optimal near-rank-1 submatrix 
        best_error: optimal rank-1 approximation error for the output submatrxi 
        best_indices: output submatrix indices 
        best_approximation_loss: optimal near-rank-1 submatrix objective   
        '''
        
        
        print(f"Available CPU cores: {os.cpu_count()}")


        best_approximation_loss = float("inf")
        best_approximation = None
        best_error = None
        best_indices = (None, None)

     
        def run_task():
            """Wrapper function to call self.run() and handle exceptions."""
            try:
                return self.run()  # returns (rank_one_approximation, error, indices, loss_value)
            except:
                print("Error - skip")
                return None

      
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(partial(self.run), range(N_rep)))

        max_error = float("-inf")  
        max_size = float("-inf")   

        for element in results:
            max_error = max(max_error, element[1])
            max_size = max(max_size, element[3]) 
            
        for result in results:
            rank_one_approximation, error, indices, size = result
            loss_value = error / max_error - self.lambd * (size / max_size) 
            if loss_value < best_approximation_loss:
                best_approximation_loss = loss_value
                best_approximation = rank_one_approximation
                best_error = error
                best_indices = indices


        return [best_approximation, best_error, best_indices, best_approximation_loss]

            

        
            






