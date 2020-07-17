import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plants_test=h5py.File('data/maize_disease_val.hdf5','r')
X_test=plants_test['features']
y_test=plants_test['labels'][:]
tmat=np.zeros((2,38))
tmat[0,10]=1
tmat[1,9]=1
y_test=y_test.dot(tmat)

def names(index):
    class_names={'Apple___Apple_scab': 0, 
                    'Apple___Black_rot': 1, 
                    'Apple___Cedar_apple_rust': 2, 
                    'Apple___healthy': 3, 
                    'Blueberry___healthy': 4, 
                    'Cherry_(including_sour)___Powdery_mildew': 5, 
                    'Cherry_(including_sour)___healthy': 6, 
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 
                    'Corn_(maize)___Common_rust_': 8, 
                    'Corn_(maize)___Northern_Leaf_Blight': 9, 
                    'Corn_(maize)___healthy': 10, 
                    'Grape___Black_rot': 11, 
                    'Grape___Esca_(Black_Measles)': 12, 
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 
                    'Grape___healthy': 14, 
                    'Orange___Haunglongbing_(Citrus_greening)': 15, 
                    'Peach___Bacterial_spot': 16, 
                    'Peach___healthy': 17, 
                    'Pepper,_bell___Bacterial_spot': 18, 
                    'Pepper,_bell___healthy': 19, 
                    'Potato___Early_blight': 20, 
                    'Potato___Late_blight': 21, 
                    'Potato___healthy': 22, 
                    'Raspberry___healthy': 23, 
                    'Soybean___healthy': 24, 
                    'Squash___Powdery_mildew': 25, 
                    'Strawberry___Leaf_scorch': 26, 
                    'Strawberry___healthy': 27, 
                    'Tomato___Bacterial_spot': 28, 
                    'Tomato___Early_blight': 29, 
                    'Tomato___Late_blight': 30, 
                    'Tomato___Leaf_Mold': 31, 
                    'Tomato___Septoria_leaf_spot': 32, 
                    'Tomato___Spider_mites Two-spotted_spider_mite': 33, 
                    'Tomato___Target_Spot': 34, 
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 
                    'Tomato___Tomato_mosaic_virus': 36, 
                    'Tomato___healthy': 37}
    name=''
    for (k,v) in class_names.items():
        if v==index:
            name=' '.join(k.replace('_', ' ').split())
    return name

def plot_boxplot(data_file,ax):
    file=h5py.File(data_file,'r')
    samples=file['predict_samples']
    class1=samples[:,768:,:]
    class1=class1.reshape(-1,class1.shape[-1])
    class_names=[names(i) for i in range(38)]
    df=pd.DataFrame(class1,columns=class_names)
    print(df.describe())
    chart=sns.boxplot(x="value", y="variable", data=pd.melt(df),color="c",linewidth=0.3,fliersize=0.05,ax=ax)
    ax.set_xscale('log')
    file.close()

def plot_entropy(y_test,data_file,ax):
    file=h5py.File(data_file,'r')
    samples=file['predict_samples']
    data=samples[:,:768,:]
    y_test=y_test[:768,:]
    n_data=data.shape[0]
    cross_entropy=np.zeros(n_data)
    for i in range(n_data):
        class1=data[i,:,:]
        cross_entropy[i]=-1.0*np.sum(np.multiply(y_test,np.log2(class1)))
    #chart=sns.distplot(cross_entropy,ax=ax)
    chart=sns.lineplot(x=range(len(cross_entropy)),y=cross_entropy,ax=ax)
    file.close()

plt.style.use('ggplot')
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(7, 5))
plot_entropy(y_test,'results/mcdropout/outofsample_output.hdf5',ax=ax0)
plot_entropy(y_test,'results/sgld/outofsample_output.hdf5',ax=ax1)
sns.despine(trim=True, left=True)   
fig.savefig('maize_disease_entropy.png', transparent=False, dpi=120, bbox_inches="tight")