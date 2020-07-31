#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

 


# # Importing the dataset

# In[5]:


dataset = pd.read_csv("heart.csv")


# In[6]:


dataset


# In[7]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])


# In[8]:


dataset['target']


# In[9]:


#gorup all similar targets 
dataset.groupby('target').size()


# In[10]:


#check shape of dataset
dataset.shape


# In[11]:


#cehck size of dataset
dataset.size


# In[12]:


#
dataset.describe()


# In[13]:


#check for null value
dataset.info()


# # Data Visualization

# In[14]:


#with pandas built-in visualization
dataset.hist(figsize=(14,14))
plt.show()


# In[29]:


plt.bar(x=dataset['sex'],height=dataset['age'])
plt.show()


# In[26]:


sns.barplot(x="fbs", y="target", data=dataset)
plt.show()


# In[27]:


sns.barplot(x=dataset['sex'],y=dataset['age'],hue=dataset['target'])


# In[31]:


sns.barplot(dataset["cp"],dataset['target'])


# In[32]:


sns.barplot(dataset["sex"],dataset['target'])


# In[34]:


sns.distplot(dataset["thal"])


# In[35]:


sns.distplot(dataset["chol"])


# In[36]:


sns.pairplot(dataset,hue='target')


# In[15]:


dataset


# In[16]:


numeric_columns=['trestbps','chol','thalach','age','oldpeak']


# In[17]:


#visualizing numeric columns
sns.pairplot(dataset[numeric_columns])


# In[40]:


y = dataset["target"]

sns.countplot(y)

target_temp = dataset.target.value_counts()

print(target_temp)


# In[41]:


# create a correlation heatmap
sns.heatmap(dataset[numeric_columns].corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# In[42]:


# create four distplots
plt.figure(figsize=(12,10))
plt.subplot(221)
sns.distplot(dataset[dataset['target']==0].age)
plt.title('Age of patients without heart disease')
plt.subplot(222)
sns.distplot(dataset[dataset['target']==1].age)
plt.title('Age of patients with heart disease')
plt.subplot(223)
sns.distplot(dataset[dataset['target']==0].thalach )
plt.title('Max heart rate of patients without heart disease')
plt.subplot(224)
sns.distplot(dataset[dataset['target']==1].thalach )
plt.title('Max heart rate of patients with heart disease')
plt.show()


# In[43]:


plt.figure(figsize=(13,6))
plt.subplot(121)
sns.violinplot(x="target", y="thalach", data=dataset, inner=None)
sns.swarmplot(x="target", y="thalach", data=dataset, color='w', alpha=0.5)


plt.subplot(122)
sns.swarmplot(x="target", y="thalach", data=dataset)
plt.show()


# In[44]:


# create pairplot and two barplots
plt.figure(figsize=(16,6))
plt.subplot(131)
sns.pointplot(x="sex", y="target", hue='cp', data=dataset)
plt.legend(['male = 1', 'female = 0'])
plt.subplot(132)
sns.barplot(x="exang", y="target", data=dataset)
plt.legend(['yes = 1', 'no = 0'])
plt.subplot(133)
sns.countplot(x="slope", hue='target', data=dataset)
plt.show()


# # Separating the Target Variable

# In[91]:


#storing the data in X and y

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[92]:


print(X)


# In[93]:


print(y)


# In[94]:


X.shape


# In[95]:


y.shape


# # Splitting the dataset into training set and testing set

# In[129]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1, shuffle = True)


# In[130]:


X_train


# In[131]:


X_test


# In[132]:


y_train


# In[133]:


y_test


# # Feature Scaling

# In[101]:


'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''


# In[102]:


print(X_train)


# In[103]:


print(X_test)


# # Training the decision tree classification model on training set
# 

# In[134]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)


# # Predicting the test set results

# In[135]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# # Making the Confusion Matrix

# In[173]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
dt_accuracy = accuracy_score(y_test, y_pred)
print(dt_accuracy)


# In[137]:


print("Accuracy on training set: {:.3f}".format(classifier.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))


# In[138]:


y_test


# In[109]:


y_pred


# In[110]:


classifier.feature_importances_


# In[90]:


def plot_feature_importances(model):
    plt.figure(figsize=(8,6))
    n_features = 13
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1,n_features)
plot_feature_importances(classifier)
#plt.savefig('feature_importance')


# In[ ]:





# # Trainning the KNN 

# For KNN first scale down the value of x and y

# In[125]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# In[127]:


print(X_train_std)


# In[128]:


print(X_test_std)


# In[167]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 22, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train_std, y_train)


# In[168]:


y_pred_knn = knn_classifier.predict(X_test_std)
print(np.concatenate((y_pred_knn.reshape(len(y_pred_knn),1), y_test.reshape(len(y_test),1)),1))


# In[174]:


from sklearn.metrics import confusion_matrix, accuracy_score
knn_cm = confusion_matrix(y_test, y_pred_knn)
print(cm)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(knn_accuracy)


# In[ ]:





# In[170]:


print("Accuracy on training set: {:.3f}".format(knn_classifier.score(X_train_std, y_train)))
print("Accuracy on test set: {:.3f}".format(knn_classifier.score(X_test_std, y_test)))


# In[ ]:





# Let's Find out the optimal number of neighbours

# In[164]:


k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    prediction_knn=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))


# In[165]:


scores


# In[166]:


plt.plot(k_range, scores_list)


# 

# In[175]:





# In[176]:





# # Training the Random Forest Classifier

# In[191]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train_std, y_train)


# In[192]:


y_pred_rf = rf_classifier.predict(X_test_std)
print(np.concatenate((y_pred_rf.reshape(len(y_pred_rf),1), y_test.reshape(len(y_test),1)),1))


# In[193]:


from sklearn.metrics import confusion_matrix, accuracy_score
rf_cm = confusion_matrix(y_test, y_pred_rf)
print(cm)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(rf_accuracy)


# In[ ]:





# Comparing the result of KNN and Decision tree 

# In[194]:


algorithms=['Decision Tree','KNN','RANDOM FOREST']
scores=[dt_accuracy,knn_accuracy,rf_accuracy]


# In[195]:


sns.set(rc={'figure.figsize':(15,7)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# In[ ]:




