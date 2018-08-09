# hamiltonian_montecarlo
Repository for Hamiltonian Monte Carlo

Dependencies : 

    Numpy
    HDF5

Optional :

    Keras
    skcuda

Get data from :

    cd data/
    # original MNIST dataset in HDF5 format
    wget http://www.geoespacial.ucm.cl/data/mnist.tar.gz
    tar -zxvf mnist.tar.gz
    # Convolutional features from the Plant Village Dataset 
    # https://github.com/spMohanty/PlantVillage-Dataset
    wget http://www.geoespacial.ucm.cl/data/plants.tar.gz
    tar -zxvf plants.tar.gz
    
Run

    python test_mnist_sgd.py

    python test_plants_sgd.py
