import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal, Empirical,Categorical
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

eta=1e-2
batch_size=10
alpha=1e-2
scaler = StandardScaler()

iris = datasets.load_iris()
data = iris.data  
labels = iris.target
classes=np.unique(iris.target)
X, y = iris.data, iris.target
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
D=X_train.shape[1]
num_classes=len(classes)
niter=1e4

# MODEL
D = X_train.shape[1]   # number of features.
num_examples = X_train.shape[0]
K = num_classes   # number of classes.
p_samples=30
epoch = 100
n_samples=niter
friction=1.0
step_size = 1e-4

x = tf.placeholder(tf.float32, [None, D])
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
y = Categorical(tf.matmul(x,w)+b)

qw= Empirical(params=tf.Variable(tf.random_normal([niter,D,K])))
qb= Empirical(params=tf.Variable(tf.random_normal([niter,K])))
      
y_ph = tf.placeholder(tf.int32, [num_examples])
inference = ed.SGHMC({w: qw, b: qb}, data={y:y_ph})


inference.initialize(n_iter=niter, n_print=n_samples, scale={y: num_examples},step_size=step_size,friction=friction)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for k in range(epoch):
    print "Epoch: %d" %(k)
    info_dict = inference.update(feed_dict={x: X_batch.values, y_ph: Y_batch.values.flatten(),keep_prob:d_rate})
    inference.print_progress(info_dict)


