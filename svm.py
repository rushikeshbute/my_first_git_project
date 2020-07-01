
import pandas as pd  
import numpy as np  
from sklearn import preprocessing, neighbors,svm
from mlxtend.plotting import plot_decision_regions



bankdata = pd.read_csv("C:/Users/HP/Desktop/NNFL/SVM.csv")

bankdata.shape
bankdata.head(2)

P = bankdata.drop('c', axis=1)  
Q = bankdata['c']  

from sklearn.model_selection import train_test_split  
P_train, P_test, Q_train, Q_test = train_test_split(P, Q, test_size = 0.25)

from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(P_train, Q_train) 

Q_pred = svclassifier.predict(P_test) 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(Q_test,Q_pred))  
print(classification_report(Q_test,Q_pred))

confidence = svclassifier.score(P_test, Q_test)
print(confidence)

plot_decision_regions(X=P.values, 
                      y=Q.values,
                      clf=svclassifier, 
                      legend=2)

print("prediction of (5,0),(-5,0),(2,5),(0,-2) ")
d={'X':[5,-5,2,0],'Y':[0,0,5,-2]};
df=pd.DataFrame(data=d)
q_pred=svclassifier.predict(df)
print(q_pred)