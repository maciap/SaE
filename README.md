# SaE
This code is submitted for the purpose of implementing the method introduced in "Sample and Expand: Discovering Low-rank Submatrices With Quality Guarantees". The code is written in Python 3. 

The current code is a **prototype**. Updates and improvements will be made soon. 


Minimal example. 
```python
from discover_near_rank_one_submatrix import SamplingAlgorithm as SamplingAlgorithmRankOne
D = np.random.randn(250, 250) # Full-rank 250 x 250 matrix with i.i.d standard gaussian entries 
delta = 0.05 # tolerance 
delta_rectangle = 1e-8
SaE_rankone = SamplingAlgorithmRankOne(D, delta=delta, delta_rectangle=delta_rectangle, 
approximate_biclique = False, sparsity_constraint = False)
output =  SaE_rankone.run()
output_submatrix_approximation = output[0]
print(f"The output submatrix has dimensions {output[0].shape}")
'''

The _notebooks_ folder contains other two examples. 
