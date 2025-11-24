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

The project contains a `bnsl` package, that contains all the algorithm implementations and necessary helpers.
```
bn-structure-learning/
├── networks/  # Original Bayesian network structures
│   ├── asia.bif
│   ├── child.bif
│   └── ...
│
├── data/
│   ├── datasets/   # Generated datasets (.dat) from sampling
│   │   ├── asia_1000.dat
│   │   ├── child_5000.dat
│   │   └── ...
│   │
│   ├── local_scores/  # Local score files (.jaa) computed by pygobnilp
│   │   ├── asia_1000.jaa
│   │   ├── child_5000.jaa
│   │   └── ...
│   │
│   └── results/  # Experiment results (.txt)
│       ├── approximation_algorithm/
│       │   ├── asia_k_4_l_2_1000_results.txt
│       │   └── ...
│       ├── partial_order_approach/
│       │   ├── child_m_4_p_2_5000_results.txt
│       │   └── ...
│       └── silander_myllymaki/
│           └── ...
│
├── src/
│   └── bnsl/  # Main package with algorithm implementations
│       ├── ..
│
├── experiments/
│   ├── notebooks/
│   │   ├── example_notebook.ipynb
│   └── configs/
│       ├── example_config.yaml
│       └── ...
│
├── pyproject.toml
├── README.md
└── .gitignore
```

### Implementation
The following algorithms are located in the  `bnsl.algorithms`-module:

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

### Data and Scores
This project **does not** implement score computation directly, instead we rely on [`pygobnilp`](https://bitbucket.org/jamescussens/pygobnilp/src/master/) to calculate local scores. All scoring related logic can be found in  `src/bnsl/scoring.py`. To change scoring method refer to this file. Pygobnilp uses **Jaakkola local-scores files** for storing local scores, see the section [Interpreting Jaakkola local-scores file](#interpreting-jaakkola-local-scores-file-jaa-files) for details. 

 The data itself can be sampled from `src/bnsl/sampling.py`. The function `sample_data` takes the path to a .bif network, and generates `n_samples` from this network. The data is then stored as a .dat file in `data/datasets`.
 It is also possible to use existing `.jaa` files, and thus skip sampling and local score generation.

### Running the project
Algorithms can be run in two ways: 
1. by importing the `bnsl` package and using it in a file, ex.:
    ```python
    from bnsl.algorithms.silander_myllymaki import run
    ```
    An example of this can be found in `experiments/notebooks/example_notebook.py`
2.  from the CLI entry point using a YAML-config file, ex.:

    ```bash
    bnsl /configs/<config_name>.yml
    ```

    To write the results to file, use the argument ````--write_path``` to specify a store location.
    ```bash
    bnsl experiments/configs/<config_name>.yml --write_results
    ```

    for more info on args in the entry point use
    ```bash
    bnsl --help
    ```

    An example of a config can be found in `experiments/configs/example_config.py`

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
