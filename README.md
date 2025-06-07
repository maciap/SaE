# SaE
This code implements the algorithms introduced in the paper "Sample and Expand: Discovering Low-rank Submatrices With Quality Guarantees". 
The code is written in Python 3. 


## 🔧 Install

- Create Conda environment:  
  ```bash
  conda env create -f SaE.yml
  ```
- Install the `pymbc` package by following the instructions at [https://github.com/wonghang/pymbc](https://github.com/wonghang/pymbc)



## 📁 Repository contents: 
- `discover_near_rank_one_submatrix.py` - Algorithm to discover near-rank-1 submatrices. 
- `discover_near_rank_k_submatrix.py` - Algorithm to discover near-rank-k submatrices. 
- `example_script.py` - Example script extracting the top five patterns according to the minimum between low-rankness score and size. 

- `notebooks/`
   - RecoverDenseLine.ipynb - Notebook showcasing the algorithm to discover near-rank-1 submatrices. 
   - RecoverDensePlane.ipynb - Notebook showcasing the algorithm to discover near-rank-k submatrices.  

- `data/`
  - `real_datasets/` - Real-world matrices used to assess the performance of SampleAndExpand. 
  - `synthetic_datasets/`- Example synthetic matrices generated according to the data-generating mechanism described in the paper. 
  -  `data_utils.py` - Utilities to read the datasets. 
  



## ✉️ Contacts

For questions or collaboration, feel free to reach out:

- Martino Ciaperoni – [martino.ciaperoni@sns.it](mailto:martino.ciaperoni@sns.it) or [martinociap@gmail.com](mailto:martinociap@gmail.com)


## ✏️ Minimal example. 
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
