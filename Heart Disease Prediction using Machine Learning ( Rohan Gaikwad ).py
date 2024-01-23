#!/usr/bin/env python
# coding: utf-8

# # .............................. Heart Disease Prediction Model Building ...........................

# # 

# ## ............................................................. Data Loading ..............................................................

# ### Importing Basic Libraries 

# In[1]:


# Basic Library
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")

# Machine Learning Models building Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# ### Reading CSV file

# In[2]:


# give the path of dataset file
rohan_df = pd.read_csv("C:\\Users\\admin\\Desktop\\heart.csv")           


# In[3]:


rohan_df.head(6)


# In[4]:


rohan_df.tail()


# ### Checking Null Values 

# In[5]:


rohan_df.isnull().sum()


# In[6]:


# To get entire Information about null values and Data Types 
rohan_df.info()


# ## ................................................. Data Visualization and Analysis ..............................................

# ### Heatmap

# In[7]:


# Heatmap used for Finding Correlation Among Attributes
plt.figure(figsize=(20,10))
sns.heatmap(rohan_df.corr() , annot = True , cmap = "terrain")
plt.show()


# ### Pair Plot

# In[8]:


# pair plot used for to visualize the relationship between different features and figure out any linear relation between them
sns.pairplot( data = rohan_df)
plt.show()


# ### Histogram

# In[9]:


# Histogram used for we can see shape of each feature and provides the count of number of observations in each bin.
rohan_df.hist(figsize = (12,12) , layout =(5,3))


# ### Box and Whisker Plot

# In[10]:


# Box and Whiskers Plot used for to find out outliers in dataset . 
rohan_df.plot( kind = "box" , subplots = True , layout = (5,3) , figsize = (12,12))
plt.show ()


# ## Visualize the features and their relation with the target                                                               ( Heart Disease or No Heart Disease )

# ### cat plot 

# In[11]:


# cat plot
sns.catplot( data = rohan_df , x = "sex" , y = "age" , hue = "target" , palette = "husl")
plt.show()


# ### Bar Plot 

# In[12]:


# Bar plot 
sns.barplot( data = rohan_df , x = "sex" , y = "chol" , hue = "target" , palette = "spring")
plt.show()


# In[13]:


rohan_df["sex"].value_counts()


# In[14]:


# Chest Pain Type
rohan_df["cp"].value_counts()        


# ### Count Plot

# In[15]:


# Count plot
sns.countplot( data = rohan_df , x = "cp" , hue = "target" , palette = "rocket")
plt.show()


# ### Cross Table for gen

# In[16]:


gen = pd.crosstab(rohan_df["sex"] , rohan_df["target"])
print(gen)


# In[17]:


gen.plot(kind = "bar" , stacked = True , color =["yellow" , "green"] , grid = False ) 
plt.show()


# ### Cross Table  for Chest Pain

# In[18]:


chest_pain = pd.crosstab(rohan_df["cp"] , rohan_df["target"])
print(chest_pain)


# In[19]:


chest_pain.plot( kind = "bar" , stacked = True , color = ["purple" , "blue"] , grid = False )
plt.show()


# ## ............................................... Preparing the Data for Model .....................................................

# ### Scaling The Data

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()
column_to_scale = [ "age" , "trestbps" , "chol" , "thalach" , "oldpeak"]
rohan_df[column_to_scale] = StandardScaler.fit_transform(rohan_df[column_to_scale])


# In[21]:


# After the Scaling Data
rohan_df.head()


# ###  Preparing Our Data For Training 

# In[22]:


# Preparing Our Data For Training 
X = rohan_df.drop(["target"] , axis = 1 )
y = rohan_df["target"]


# In[23]:


X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.3 , random_state = 40 )


# ### Check Sample Size

# In[24]:


print("X_train :", X_train.size)
print("X_test :", X_test.size)
print("y_train :" ,y_train.size)
print("y_test :", y_test.size)


# ### Applying  Machine Learning Algorithm : Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression()

model1 = lr.fit(X_train , y_train)
Prediction1 = model1.predict(X_test)


# In[26]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , Prediction1)
cm


# In[27]:


sns.heatmap(cm , annot = True , cmap = "BuPu")
plt.show()


# ### finding the accuracy of the model

# In[28]:


TP = cm[0][0]
TN = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print("Testing Accurancy :" , (TP+TN)/(TP+TN+FN+FP))


# In[29]:


from sklearn.metrics import accuracy_score

print( "Testing Accurancy :" , accuracy_score(y_test , Prediction1))


# ### precision and recall of the model

# In[30]:


from sklearn.metrics import classification_report
print( classification_report(y_test , Prediction1))


# ### Machine Learning Algorithm : KNeighborsClassifier

# In[31]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
model2=KNN.fit(X_train,y_train)
Prediction2 = model2.predict(X_test)


from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction2))


# ### Machine Learning Algorithm : SVC

# In[32]:


from sklearn.svm import SVC
SVC = SVC()
model3 = SVC.fit(X_train,y_train)
Prediction3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction3))


# ### Machine Learning Algorithm : DecisionTreeClassifier

# In[34]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
model4 = DT.fit(X_train,y_train)
Prediction4 = model4.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction4))


# ###  Machine Learning Algorithm : GaussianNB

# In[35]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
model5 = GNB.fit(X_train,y_train)
Prediction5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction5))


# ###  Machine Learning Algorithm : RandomForestClassifier

# In[37]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
model6 = RF.fit(X_train, y_train)
Prediction6  = model6.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction6))


# In[38]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Log-Reg', 'KNN', 'SVC', 'Des-Tree', 'Gaus-NB', 'RandomForest']
students = [92.30,85.71,90.10, 70.32 ,87.91,87.91]
ax.bar(langs,students)
plt.show()


# ## The Best Accuracy is given by Logistic Regression is 92 . Hence we will use LogisticRegression algorithms for training my model.

# ## --------------------------------------------------------------------------------------------------------------------------
