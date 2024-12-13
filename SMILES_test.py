from padelpy import from_smiles
import pandas as pd


# calculate molecular descriptors for propane
descriptors_1 = from_smiles("C(C(=O)O)Cl")

print(descriptors_1)

desc1_df = pd.DataFrame([descriptors_1])

print(desc1_df.head())
print(desc1_df.shape)

