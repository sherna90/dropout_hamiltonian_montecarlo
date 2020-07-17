import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py 
import sys 
sys.path.append('./')
import hamiltonian.utils as utils
import hamiltonian.models.gpu.softmax as base_model
import hamiltonian.inference.gpu.sgld as inference
import pickle
import seaborn as sns
from scipy import stats

eta=1e-5
epochs=100
burnin=1
batch_size=250
alpha=1./100.
data_path = './data/'

train_file='plant_village_train.hdf5'
test_file='plant_village_val.hdf5'

plants_train=h5py.File(data_path+train_file,'r')
X_train=plants_train['features']
y_train=plants_train['labels']
plants_test=h5py.File(data_path+test_file,'r')
X_test=plants_test['features']
y_test=plants_test['labels']

D=X_train.shape[1]
K=y_train.shape[1]
import time

start_p={'weights':np.zeros((D,K)),
        'bias':np.zeros((K))}
hyper_p={'alpha':alpha}

start_time=time.time()
model=base_model.softmax(hyper_p)


labels={'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3, 'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6, 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8, 'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11, 'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 'Grape___healthy': 14, 'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17, 'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20, 'Potato___Late_blight': 21, 'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24, 'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27, 'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 'Tomato___Late_blight': 30, 'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32, 'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___healthy': 37}
labels = [' '.join(l.replace("_"," ").split()) for l in labels.keys()]


def train_model():
    sampler=inference.sgld(model,start_p,step_size=eta)
    par,loss=sampler.sample(epochs=epochs,burnin=burnin,batch_size=batch_size,gamma=0.9,X_train=X_train,y_train=y_train,verbose=True)
    print('SGD, time:',time.time()-start_time)
    loss=pd.DataFrame(loss)
    loss.to_csv('loss.csv',sep=',',header=False)
    with open('model.pkl','wb') as handler:
        pickle.dump(par,handler)

def test_model():
    with open('model.pkl','rb') as handler:
        samples=pickle.load(handler)
    predict_samples=[]
    for i in range(epochs):
        print('prediction : {0}'.format(i))
        par={var:samples[var][i] for var in samples.keys()}
        y_pred=model.predict(par,X_test,prob=True,batchsize=32)
        y_pred=y_pred.reshape(-1, y_pred.shape[-1])
        predict_samples.append(y_pred)
    predict_samples=np.asarray(predict_samples)
    with h5py.File('output.hdf5', 'w') as f:
        f["predict_samples"] = predict_samples

def evaluate_model():
    file=h5py.File('results/sgld/output.hdf5', 'r')
    epochs=file['predict_samples'].shape[0]
    precision=[]
    recall=[]
    fscore=[]
    for i in range(epochs):
        y_pred=file['predict_samples'][i,:,:]
        report=classification_report(y_test[:].argmax(axis=1), y_pred.argmax(axis=1),output_dict=True)
        precision.append(report['weighted avg']['precision'])
        recall.append(report['weighted avg']['recall'])
        fscore.append(report['weighted avg']['f1-score'])
    results=pd.DataFrame({'precision':precision,'recall':recall,'f1':fscore})
    print(results.describe())

def test_outofsample():
    plants_test=h5py.File(data_path+'maize_disease_val.hdf5','r')
    X_test=plants_test['features']
    y_test=plants_test['labels'][:]
    tmat=np.zeros((2,38))
    tmat[0,10]=1
    tmat[1,9]=1
    y_test=y_test.dot(tmat)
    with open('results/sgld/model.pkl','rb') as handler:
        samples=pickle.load(handler)
    predict_samples=[]
    conf_mat=[]
    precision=[]
    recall=[]
    fscore=[]
    for i in range(epochs):
        print('prediction : {0}'.format(i))
        par={var:samples[var][i] for var in samples.keys()}
        y_pred=model.predict(par,X_test,prob=True,batchsize=y_test.shape[0])
        y_pred=y_pred.reshape(-1, y_pred.shape[-1])
        conf_mat.append(confusion_matrix(y_test[:].argmax(axis=1), y_pred.argmax(axis=1),labels=np.arange(38))[9:11,:])
        report=classification_report(y_test[:].argmax(axis=1), y_pred.argmax(axis=1),output_dict=True)
        precision.append(report['weighted avg']['precision'])
        recall.append(report['weighted avg']['recall'])
        fscore.append(report['weighted avg']['f1-score'])
        print("-----------------------------------------------------------")
        predict_samples.append(y_pred)
    results=pd.DataFrame({'precision':precision,'recall':recall,'f1':fscore})
    print(results.describe())
    with open('sgmcmc_class.pkl','wb') as handler:
        pickle.dump(np.median(np.asarray(conf_mat),axis=0),handler)
    sns.heatmap(np.median(np.asarray(conf_mat),axis=0),linewidths=.5,cmap="Blues",robust=True,yticklabels=labels[9:11],xticklabels=labels,cbar=False,square=True)
    plt.title('SGLD', fontsize=16)
    plt.savefig("sgld_class.pdf", bbox_inches="tight")
    plt.show()
    with h5py.File('results/sgld/outofsample_output.hdf5', 'w') as f:
        f.create_dataset('predict_samples', data=np.asarray(predict_samples)) 
        
def evaluate_model_outofsample():
    plants_test=h5py.File(data_path+'maize_disease_val.hdf5','r')
    X_test=plants_test['features']
    y_test=plants_test['labels'][:]
    tmat=np.zeros((2,38))
    tmat[0,9]=1
    tmat[1,10]=1
    y_test=y_test.dot(tmat)
    file=h5py.File('results/sgld/outofsample_output.hdf5', 'r')
    epochs=file['predict_samples'].shape[0]
    correct=[]
    incorrect=[]
    tp=[]
    fp=[]
    timepoint=[]
    for i in range(epochs):
        y_pred=file['predict_samples'][i,:,:]
        t_index=(y_test.argmax(axis=1)==y_pred.argmax(axis=1))
        correct.append(y_pred[t_index].max(axis=1)[:])
        incorrect.append(y_pred[~t_index].max(axis=1)[:])
        tpi=y_test[t_index].argmax(axis=1)
        fpi=y_test[~t_index].argmax(axis=1)
        fp=fp+['healthy' if c==10 else 'non-healthy' for c in fpi]
        tp=tp+['healthy' if c==10 else 'non-healthy' for c in tpi]
        timepoint.append([i]*y_pred.shape[0])
    timepoint=np.concatenate(timepoint)
    correct=np.concatenate(correct)
    label0=['correct']*correct.shape[0]
    incorrect=np.concatenate(incorrect)
    label1=['incorrect']*incorrect.shape[0]
    results=pd.DataFrame({'class':tp+fp,'predicted label':np.concatenate([label0,label1]),'probability':np.concatenate([correct,incorrect]),'timepoint':timepoint})
    non_healthy_correct=results[(results['class']=='non-healthy') & (results['predicted label']=='correct')]['probability'].values
    non_healthy_incorrect=results[(results['class']=='non-healthy') & (results['predicted label']=='incorrect')]['probability'].values
    healthy_correct=results[(results['class']=='healthy') & (results['predicted label']=='correct')]['probability'].values
    healthy_incorrect=results[(results['class']=='healthy') & (results['predicted label']=='incorrect')]['probability'].values
    print('correct healthy, mean : %5.2f : , std : %5.2f'%(np.mean(healthy_correct),np.std(healthy_correct)))
    print('incorrect healthy, mean : %5.2f : , std : %5.2f'%(np.mean(healthy_incorrect),np.std(healthy_incorrect)))
    print('p-value healthy : %5.5f'%(stats.ttest_ind(healthy_correct,healthy_incorrect)[1]))
    print('correct non-healthy, mean : %5.2f : , std : %5.2f'%(np.mean(non_healthy_correct),np.std(non_healthy_correct)))
    print('correct non-healthy, mean : %5.2f : , std : %5.2f'%(np.mean(non_healthy_incorrect),np.std(non_healthy_incorrect)))
    print('p-value non-healthy : %5.5f'%(stats.ttest_ind(non_healthy_correct,non_healthy_incorrect)[1]))
    #sns.lmplot(x="timepoint", y="probability", hue='class', col='label',data=results)
    sns.catplot(x="predicted label", y="probability", kind="box", hue='class',data=results)
    plt.title('SGLD', fontsize=16)
    plt.savefig("sgld_out.pdf", bbox_inches="tight")
    plt.show()

#train_model()
#test_model()
#test_outofsample()
evaluate_model_outofsample()

plants_train.close()
plants_test.close()
