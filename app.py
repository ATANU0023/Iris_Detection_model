'''go to terminal and install
pip install numpy 
            matplotlib 
            seaborn 
            pandas 
            scikit-learn
'''





# iris flower classification
#import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


columns = ['sepal_length','sepal_width','petal_length','petal_width','class_labels']
#load the data

df = pd.read_csv('iris.data',names=columns)
df.head()        


# analyze and visualize the dataset
# some basic statistical analysis about the data
df.describe()

#visualize the whole dataset
sns.pairplot(df, hue='class_labels')

#separate features and taget
data = df.values
x = data[:,0:4]
y = data[:,4]

#calculate average of each feature for all classes
y_data = np.array([np.average(x[:, i][y==j].astype('float32')) for i in range (x.shape[1])for j in (np.unique(y))])
y_data_reshaped = y_data.reshape(4,3)
y_data_reshaped = np.swapaxes(y_data_reshaped, 0, 1)
x_axis = np.arange(len(columns)-1)
width = 0.25

#plot the average
plt.bar(x_axis, y_data_reshaped[0], width, label = 'Setosa')
plt.bar(x_axis+width, y_data_reshaped[1], width, label = 'Versicolour')
plt.bar(x_axis+width*2, y_data_reshaped[2], width, label = 'Virginica')
plt.xticks(x_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()


#STEP 3 MODEL TRAINING
#split the data to train and test dataset
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


#support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(x_train, y_train)



#MODEL EVALUATION
#predict from the test dataset
predictions = svn.predict(x_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


'''# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

                    precision    recall  f1-score   support

    Iris-setosa         1.00      1.00      1.00         9
Iris-versicolor         1.00      0.83      0.91        12
 Iris-virginica         0.82      1.00      0.90         9

       accuracy                                 0.93        30
      macro avg         0.94      0.94      0.94        30
   weighted avg         0.95      0.93      0.93        30'''


x_new = np.array([[3, 2, 1, 0.2],[ 4.9, 2.2, 3.8, 1.1], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(x_new)
print("Prediction of Species: {}".format(prediction))


#save the model
import pickle
with open('SVM.pickle','wb') as f:
    pickle.dump(svn, f)

#load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)
model.predict(x_new)        