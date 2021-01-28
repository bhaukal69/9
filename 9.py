from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris_dataset=load_iris()
X_train,X_test,Y_train,Y_test=train_test_split(iris_dataset["data"],iris_dataset["target"])
kn=KNeighborsClassifier()
kn.fit(X_train,Y_train)
y_pred=kn.predict(X_test)

a=[0]*2
true=0
for i,j in zip(Y_test,y_pred):
    if i==j:
        a[0]=[i,j,'Correct']
        true+=1
    else:
        a[1]=[i,j,'Wrong']
print("\nAccuracy",true/len(y_pred)*100)
print(a)
