# bn-structure-learning

Implementations of different BN structure learning algorithms.

## Getting Started
The project requires that you have [`uv`](https://docs.astral.sh/uv/) installed.

1. Create a virtual environment: 
    ```bash
    uv venv
    ```

2. Install requirements:

    ```bash
    uv sync --all-extras --all-packages --all-groups
    source .venv/bin/activate
    ```

3. (Optional) Editable install for development.

    If you plan to make changes inside the `bnsl` package, install it in editable mode:
    ```bash
    uv pip install -e .
    ```

## Project Structure

This project **does not** implement score computation directly, instead we rely on [`pygobnilp`](https://bitbucket.org/jamescussens/pygobnilp/src/master/) to calculate local scores.

The project contains a `bnsl` package, that contains all the algorithm implementations and necessary helpers.

### Data and Scores
 The algorithms in this project assume that **local scores** for your dataset are already available.  
These must be stored as **Jaakkola local-scores files** inside the `data/local_scores` folder. See the section [Interpreting Jaakkola local-scores file](#interpreting-jaakkola-local-scores-file-jaa-files) for details.

To generate these files, run: `scripts/write_local_scores.py` pointing it to the dataset you want to process. This will create the `data/local_scores/` folder (if it does not exist) and write the `.jaa` file for you. 

 The data itself can be retrieved in two ways.
 - either from the pygobnilp subrepo, available in the folder: `pygobnilp/data/` or custom datasets can be placed in the `data/datasets`-folder. 

- alternatively, you can generate your own data from a given bayesian network into  the `data/`-folder, use the  `scripts/sample_data.py` script. This assumes a `.bif`-file specifying the network exists in the `networks/`-folder.

### Running the project
Algorithms can be run by import the `bnsl` package and using it in a file, ex:
```python
from bnsl.algorithms.silander_myllymaki import run
```

Algorithms can also be run from the CLI entry point:
```bash
bnsl silander_myllymaki --jaa data/local_scores/local_scores_asia_10000.jaa
```

for more info on args in the entry point use
```bash
uv run bnsl --help
```

### Implementation
The following algorithms are located in the  `bnsl_algorithms`-folder:

- `silander_myllymaki.py` contains an implementation of the algorithm described in:  

    > **Silander, T. & Myllymäki, P. (2012)**  
    > *A simple approach for finding the globally optimal Bayesian network structure*.  
    > [arXiv:1206.6875](https://arxiv.org/abs/1206.6875)

- `partial_order_approach.py` contains an implementation of the algorithm described in:  

    > **Parviainen, P. & Koivisto, M. (2013)**  
    > *Finding optimal Bayesian networks using precedence constraints.*  
    > *Journal of Machine Learning Research*, **14**(1), 1387–1415.  
    > [JMLR Paper](https://www.jmlr.org/papers/v14/parviainen13a.html)

- `approximation_algorithm.py` contains an implementation of the algorithm described in:  

    > **Kundu, M., Parviainen, P. & Saurabh, S. (2024)**  
    > *Time–Approximation Trade-Offs for Learning Bayesian Networks.*  
    > *Proceedings of Machine Learning Research (PMLR)*, **246**, 486–497.  
    > [PMLR Paper](https://proceedings.mlr.press/v246/kundu24a.html)

## Interpreting Jaakkola local-scores file (.jaa files)

A `.jaa` file stores **local scores** for Bayesian network structure learning. It lists, for each variable, the score of each **parent set**. 


1. **First line**: a single integer — the **number of variables** in the dataset.
2. Then **one block per variable**:
   - **Header line**:  
     ```
     <VariableName> <K>
     ```
     where `K` is the number of parent sets listed for this variable. K is the number of candidate parent sets that survive the limits and pruning specified. To edit this, refer to the arguments of Gobnilp in `write_local_scores.py` file.
   - **K lines**, one per parent set:  
     ```
     <score> <pcount> [<Parent1> <Parent2> ...]
     ```
