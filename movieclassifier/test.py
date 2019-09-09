import pickle 
import re
import os 
import numpy as np
from vectorizer import vect 

if  __name__ == "__main__":  
    cur_dir = os.path.dirname(__file__)
    clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','classifier.pkl'),'rb')) 
    label ={0:'negative',1:'positive'}
    example = ['I love this movie'] 
    X = vect.transform(example) 
    print("Movie review: %s" % example)
    print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100)) 
    
    example = ['I hate this movie'] 
    X = vect.transform(example) 
    print("Movie review: %s" % example)
    print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))

    