# %%
import numpy as np
import pandas as pd
from mcp import Mcp
from sklearn import datasets

#  %%
rawData = datasets.load_breast_cancer()
df = pd.DataFrame(rawData["data"][:, 0:9])
df.columns = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry"]

df["result"] = rawData["target"]

# %%
