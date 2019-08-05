import matplotlib.pyplot as plt 

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(classes)
    plt.xticks(tick_marks, tick_marks, rotation=45,fontsize=8)
    plt.yticks(tick_marks, tick_marks,fontsize=8)
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()