from pgmpy.readwrite import BIFReader
import pandas as pd

NETWORK_PATH = "networks/child.bif"
WRITE_PATH = "data/datasets/child_10000.dat"

reader = BIFReader(NETWORK_PATH) 
model = reader.get_model()

# sample synthetic data
data = model.simulate(n_samples=10000)

# encode each categorical column to integers based on its unique order
encoded = data.apply(lambda col: pd.Categorical(col).codes)

# compute arities (max code + 1 for each column)
arities = (encoded.max(axis=0) + 1).astype(int).tolist()

with open(WRITE_PATH, "w", encoding="utf-8") as f:
    f.write("\t".join(encoded.columns) + "\n")     # header
    f.write("\t".join(map(str, arities)) + "\n")   # arities

encoded.to_csv(WRITE_PATH, sep="\t", index=False, header=False, mode="a")