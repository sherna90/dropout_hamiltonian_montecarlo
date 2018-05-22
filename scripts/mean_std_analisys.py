import pandas as pd
import numnpy as np

df = pd.read_csv("../build/temp_predict.csv")
arr =df.values

mean = arr.mean(axis=0)
std = arr.std(axis=0)
df_mean = pd.DataFrame(mean)
df_std = pd.DataFrame(std)

df_mean.to_csv("../build/mean.csv", ',', header=None, index = False)
df_std.to_csv("../build/std.csv", ',', header=None, index = False) 