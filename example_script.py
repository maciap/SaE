import numpy as np
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), 'datasets'))
from data_utils import read_data , normalize_array_min_max
from discover_near_rank_one_submatrix import SamplingAlgorithm as SamplingAlgorithmRankOne 
from discover_near_rank_k_submatrix import SamplingAlgorithm as SamplingAlgorithmRankK
import time 

def compute_sparsity(matrix, tol=1e-8):
    """
    Computes the sparsity level of a NumPy array with a small tolerance for zero.
    
    @parameters:
    matrix (numpy.ndarray): Input array or matrix
    tol (float): Tolerance for considering a value as zero (default: 1e-8)
    
    returns:
    sparsity level (float) from 0 to 1     
    """
    total_elements = matrix.size  # Total number of elements
    zero_elements = np.count_nonzero(np.abs(matrix) < tol)  # Count near-zero values
    sparsity_level = zero_elements / total_elements  # Compute sparsity
    return sparsity_level
    

def find_top_five_patterns_max_min(D, N_init, deltas, ranks, sparsity_constraint): 
    ''' 
    Helper function to perform experiments similar to those reported in the paper. 
    The function runs Sample And Expand many times to identify the top five patterns that maximize 
    the minimum between low-rankness score and size 

    @params: 
    D: input matrix (numpy.ndarray) 
    N_init: number of initializations per configuration delta / rank (int) 
    deltas: tolerance values to explore (list of floats) 
    ranks: target ranks to explore (list) 
    
    returns 
    best_indices: indices of top 5 submatrices 
    best_lowranknesses: low-rankness of top 5 submatrices 
    best_sizes: sizes of top 5 submatrices 
    '''

    all_approximations = []
    max_lowrankness = float("-inf")
    max_size = float("-inf") 
    all_lowrankness = [] 
    all_size = []
    all_indices = []
    n_patterns = 5 
    delta_rectangle = 1e-10 

    for delta in deltas: # Iterate over tolerance parameters 

        for rank in ranks: # Iterate over ranks 

            for rp in range(N_init): 
                
                if rank==1:  # Run Sample and Expand for target rank 1 
                    SaE_rankone = SamplingAlgorithmRankOne(D, delta=delta, delta_rectangle=delta_rectangle, 
                    approximate_biclique = "biclustering", sparsity_constraint = sparsity_constraint)
                    out =  SaE_rankone.run()
                    best_approximation , best_error , best_indices,  best_size  = out 

                else: # Run Sample and Expand for target rank k 
                    SaE_rank_k = SamplingAlgorithmRankK(D, k = rank, delta=delta, delta_rectangle=delta_rectangle, 
                    approximate_biclique = "biclustering", sparsity_constraint = sparsity_constraint)
                    out =  SaE_rank_k.run()
                    best_approximation , best_error , best_indices, best_size = out 
        
                    
                # Store submatrix indices 
                all_approximations.append(out) 
                all_indices.append(best_indices) 

                # Store size 
                size = np.prod(best_approximation.shape) 
                all_size.append(size) 

                # Store low rankness 
                singular_values = np.linalg.svd(D[np.ix_(best_indices[0], best_indices[1])], full_matrices=False, compute_uv=False)                    
                low_rankeness  = (singular_values[0] ** 2) / np.sum(singular_values ** 2)
                all_lowrankness.append(low_rankeness)


    # Normalize the size and low-rankness 
    avg_lrn = np.mean(all_lowrankness)
    std_lrn = np.std(all_lowrankness)
    normalized_low_rankeness = (all_lowrankness - avg_lrn) / std_lrn

    avg_size = np.mean(all_size)
    std_size = np.std(all_size)
    normalized_sizes =  (all_size - avg_size) / std_size

    # Compute minimum between low-rankness and size 
    all_scores = []
    for i in range(len(all_approximations)):
        low_rankeness = normalized_low_rankeness[i] 
        size = normalized_sizes[i] 
        score = min(  low_rankeness ,  size )
        all_scores.append(score) 


    # Find top 5 patterns according to the score definition: normalize minimum between low-rankness and size 
    top_5_indices = np.argsort(all_scores)[-n_patterns:][::-1]  # Sort in descending order 
    best_indices = [all_indices[j] for j in top_5_indices] 
    best_lowranknesses = [all_lowrankness[h] for h in top_5_indices] 
    best_sizes = [all_size[d] for d in top_5_indices]        
    
    return best_indices , best_lowranknesses, best_sizes


if __name__ == '__main__': 

    if len(sys.argv) < 5:
        print("Usage: python script.py <dataset_name> <N_init> <deltas> <ranks>")
        print("Example: python script.py cameraman 10 0.1,0.2,0.3 1,2,3")
        sys.exit(1)

    dataset_name = sys.argv[1]  
    N_init = int(sys.argv[2])  # convert to int
    deltas = [float(d) for d in sys.argv[3].split(',')]
    ranks = [int(r) for r in sys.argv[4].split(',')]


    D = read_data(dataset_name, base_path="datasets/real_datasets/").astype(float) 
    D_normalized = normalize_array_min_max(D) # Normalize data in [0,1] 
    sparsity_level = compute_sparsity(D_normalized) # Compute sparsity 

    if sparsity_level > 0.2: # Set the sparsity constraint based on threshold (0.2) 
        sparsity_constraint = True 
    else: 
        sparsity_constraint = False 

    # Run experiment 
    start_time = time.time() 
    best_indices , best_lowranknesses, best_sizes = find_top_five_patterns_max_min(D_normalized, N_init, deltas, ranks, sparsity_constraint) 
    print(f"Experiment terminated in {round(time.time() - start_time, 2)} seconds.")




    

    

