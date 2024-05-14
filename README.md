![330041292-bc0b8b0d-184b-4198-ba03-66b8a34c670f](https://github.com/Gajalakshmivelmurugan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144871940/cc4f72ff-01e5-492a-8998-5a4b6e4b3277)# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the program
2.Import the python pandas library as pd
3.Read the contents of the Spam csv file
4.Display the first 5 rows of the dataset using head()
5.Assign x as v1 values and y as v2 values
6.From sklearn library select the feature extraction and import CountVectorizer
7.CountVectorizer will convert the Text to Numerical Data
8.From sklearn library import Support Vector Classifier (ie. SVC)
9.Predict the x_test using SVC
10.Print the accuracy of the SVM Model 11.Stop the program 
```
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: gajalakshmi V
RegisterNumber:  212223040047
*/
import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding = 'Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Result output

![330040985-f6d81040-672e-4f3f-ba9c-e7b799d8c12b](https://github.com/Gajalakshmivelmurugan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144871940/58b38d24-1629-420d-9039-68032335d37c)

data.head()

![330041081-9e237ba6-faf8-4531-80a6-fd8d498714bb](https://github.com/Gajalakshmivelmurugan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144871940/a7450ecd-a456-47d0-b4b0-3598f977cd04)

data.info()

![330041160-8a4eb0b3-35e6-40cb-a4d9-eef2ef716be6](https://github.com/Gajalakshmivelmurugan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144871940/16b043f6-f7a2-49de-9354-010d34b9e808)

data.isnull().sum()

![330041212-8a6c0be7-5a4a-4193-97fd-ffcc2909a7f6](https://github.com/Gajalakshmivelmurugan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144871940/e91396d7-0fca-4b8e-b96b-a8dfe52cb7c4)

Y_prediction value

![330041292-bc0b8b0d-184b-4198-ba03-66b8a34c670f](https://github.com/Gajalakshmivelmurugan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144871940/034f4eed-6395-4749-8871-2f2fa76625a9)

Accuracy value

![330041322-21a27655-5d61-4af2-9138-aa153437806c](https://github.com/Gajalakshmivelmurugan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144871940/a965815e-25c1-4342-9729-6c8910bba240)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
