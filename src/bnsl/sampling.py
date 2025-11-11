from pgmpy.readwrite import BIFReader
import pandas as pd
import os

def sample_data(network_path:str, n_samples:int, seed:int)-> str:
    """Samples data from a Bayesian network in BIF format and writes it to a .dat file.
    Saves the data as data/datasets/{network_name}_{n_samples}.dat
    network_path: Path to the BIF file of the Bayesian network.
    n_samples: Number of samples to generate.
    returns: Path to the generated .dat file.
    """
    reader = BIFReader(network_path)
    model = reader.get_model()

    network_name = os.path.splitext(os.path.basename(network_path))[0]
    write_path = f"data/datasets/{network_name}_{n_samples}.dat"

    if os.path.exists(write_path):
        print(f"File already exists: {write_path}. Skipped sampling.")
        return write_path
    
    os.makedirs(os.path.dirname(write_path), exist_ok=True)

    state_names: dict[str, list[str]] = {}
    for cpd in model.get_cpds():
        state_names[cpd.variable] = list(cpd.state_names[cpd.variable])

    # get the state names (categorical values) and node order defined in the model
    variables= list(model.nodes())
    df = model.simulate(n_samples=n_samples, seed=seed)[variables]
    
    # encode using the extracted ordering
    encoded = pd.DataFrame(index=df.index)
    for var in variables:
        cats = state_names[var]
        encoded[var] = pd.Categorical(df[var], categories=cats, ordered=True).codes

        # sanity checks
    if (encoded == -1).any().any():
        bad = {col: encoded[col].eq(-1).sum() for col in variables if encoded[col].eq(-1).any()}
        raise ValueError(f"Found values outside declared categories (coded as -1): {bad}")
    
    encoded = encoded.astype(int)

    # get arities from the model (arity of a variable = how many distinct values it can take)
    card = model.get_cardinality()
    arities = [int(card[var]) for var in variables]

    with open(write_path, "w", encoding="utf-8") as f:
        f.write(" ".join(encoded.columns) + "\n")
        f.write(" ".join(map(str, arities)) + "\n")
    encoded.to_csv(write_path, sep=" ", index=False, header=False, mode="a")

    return write_path