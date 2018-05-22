import pandas as pd
#path = "hmc3/hmc_"
path = "../build/mdhmc mnist 0.9/"


mean = pd.read_csv(path+"predict_proba_mean.csv", header = None, names =['mean 0', 'mean 2', 'mean 5', 'mean 8', 'mean 9', 'mean 4', 'mean 7', 'mean 3', 'mean 6', 'mean 1'])
std = pd.read_csv(path+"predict_proba_std.csv", header = None, names =['std 0', 'std 2', 'std 5', 'std 8', 'std 9', 'std 4', 'std 7', 'std 3', 'std 6', 'std 1'])
max_mean = pd.read_csv(path+"predict_proba_max.csv", header = None, names =['max'])
gt = pd.read_csv(path+"Y_test.csv", header = None, names =['GT'])

result = pd.concat([mean, max_mean, gt, std], axis=1, join_axes=[mean.index])

result.to_csv(path+"result.csv",sep=',', encoding='utf-8')