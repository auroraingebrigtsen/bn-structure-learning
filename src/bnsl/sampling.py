from pgmpy.readwrite import BIFReader
import pandas as pd
import os

def sample_data(network_path:str, n_samples:int=10000)-> str:
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

    df = model.simulate(n_samples=n_samples)

    # get the state names (categorical values) defined in the model
    state_names = model.states
    
    # encode using the model's defined ordering
    encoded = df.copy()
    for col in df.columns:
        # create categorical with explicit categories from the model
        encoded[col] = pd.Categorical(df[col], categories=state_names[col]).codes

    # get arities from the model (arity of a variable = how many distinct values it can take)
    card = model.get_cardinality()
    arities = [int(card[var]) for var in encoded.columns]

    with open(write_path, "w", encoding="utf-8") as f:
        f.write(" ".join(encoded.columns) + "\n")
        f.write(" ".join(map(str, arities)) + "\n")
    encoded.to_csv(write_path, sep=" ", index=False, header=False, mode="a")

    return write_path