import numpy as np

D = 2

centers = [np.random.random_integers(0,10,D)]
print(centers)

centers = []
for i in range(3):
    centers.append(np.random.random_integers(0,10,D))

print(centers)

aux = [np.random.random_integers(0,10,D) for i in range(3)]
print(aux)

