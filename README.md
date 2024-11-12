# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1:Start the program

Step 2:Import the required library and read the dataframe.

Step 3:Write a function computeCost to generate the cost function.

Step 4:Perform iterations og gradient steps with learning rate.

Step 5:Plot the Cost function using Gradient Descent and generate the required graph.

Step 6:End the program
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: gajalakshmi V
RegisterNumber:  212223040047
*/
import pandas as pd
data = pd.read_csv("C:/Users/SEC/Downloads/spam.csv" ,encoding = "Windows-1252")
from sklearn.model_selection import train_test_split
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.35,random_state = 42)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

![383420060-d85cd853-6e2c-4e19-96ec-212a70fdb836](https://github.com/user-attachments/assets/a2211e1e-3965-494b-bdbb-170e11e69645)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
