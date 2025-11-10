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

    # Obtain the name of the network from the path
    network_name = network_path.split("/")[-1].split(".")[0]
    write_path = f"data/datasets/{network_name}_{n_samples}.dat"

    # If the file already exists, skip sampling
    if os.path.exists(write_path):
        print(f"File already exists: {write_path}. Skipped sampling.")
        return write_path
    
    os.makedirs(os.path.dirname(write_path), exist_ok=True)

    # sample synthetic data
    data = model.simulate(n_samples=n_samples)
    # encode each categorical column to integers based on its unique order
    encoded = data.apply(lambda col: pd.Categorical(col).codes)

    # compute arities (max code + 1 for each column)
    arities = (encoded.max(axis=0) + 1).astype(int).tolist()

    with open(write_path, "w", encoding="utf-8") as f:
        f.write("\t".join(encoded.columns) + "\n")     # header
        f.write("\t".join(map(str, arities)) + "\n")   # arities


    encoded.to_csv(write_path, sep="\t", index=False, header=False, mode="a")

    return write_path