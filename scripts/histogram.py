import matplotlib.pyplot as plt
import pandas as pd
path = "../build/mdhmc mnist 0.9/"

data = pd.read_csv(path+"histogram.csv", header = None)
data.hist()
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.savefig(path+"histogram.png")
plt.show()
