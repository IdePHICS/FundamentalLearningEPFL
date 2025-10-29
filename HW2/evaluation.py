# import the abstract model of sklearn
from sklearn.base import BaseEstimator
import numpy as np

def load_file(file):
    with np.load('data/'+file+'.npz') as data:
        X = data['X']
        y = data['y']
    return X, y

X, y = load_file('test')

X = X.reshape(X.shape[0], -1)

def evaluate(model: BaseEstimator):
    # check if model is an instance of BaseEstimator
    if not isinstance(model, BaseEstimator):
        print("Evaluation failed! The provided model is not an sklearn model")
        return
    
    # chek if the model has a predict method
    if not hasattr(model, 'predict'):
        print("Evaluation failed! The provided model does not have a predict method")
        return

    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print("All good, you are ready for HomeWork submission! The accuracy on 'train' is: ", accuracy)
    
    