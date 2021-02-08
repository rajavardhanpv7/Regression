#!/usr/bin/env python
# coding: utf-8

# Univariate Analyis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Ridge and Lasso Regression

# In[2]:


from sklearn.datasets import load_boston


# In[3]:


df=load_boston()


# In[4]:


df


# In[5]:


dataset=pd.DataFrame(df.data)


# In[6]:


print(dataset.head())


# In[7]:


dataset.columns=df.feature_names


# In[8]:


dataset.head()


# In[9]:


df.target.shape


# In[10]:


dataset['price']=df.target


# In[11]:


dataset.head()


# In[12]:



x=dataset.iloc[:,:-1] # independent features
y=dataset.iloc[:,-1]#dependent features


# linear Regression

# In[13]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,x,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# Ridge Regression

# In[14]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[15]:


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-6,1e-3,1,2,3,4,5,10,15,20,25,45,50,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)


# In[16]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[17]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


# In[18]:


lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-6,1e-3,1,2,3,4,5,10,15,20,25,45,50,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(x,y)


# In[19]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split (x,y,test_size=0.3,random_state=0) 
prediction_lasso=lasso_regressor.predict(x_test)
prediction_ridge=ridge_regressor.predict(x_test)


# In[21]:


import seaborn as sns


# In[22]:


sns.distplot(y_test-prediction_lasso)


# In[23]:


sns.distplot(y_test-prediction_ridge)


# Linear Regression with single variable

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


df=pd.read_excel(r'C:\Users\director\Downloads\LRwithsingle.xls')


# In[26]:


df.iloc[:,:].values


# In[27]:


plt.scatter(df.area,df.price,color='red',marker='+')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(us$)')


# In[28]:


reg=linear_model.LinearRegression()


# In[29]:


reg.fit(df[['area']],df.price)


# In[30]:


reg.predict([[3300]])


# In[31]:


reg.coef_


# In[32]:


reg.intercept_


# In[33]:


# Maths behind the calculations
# price= m*area+c
# y=mx+c
89.87676056*3300+341989.43661971827


# In[34]:


D=pd.read_excel(r'C:\Users\director\Downloads\area.xls')


# In[35]:


p=reg.predict(D)


# In[36]:


D['prices']=p


# In[37]:


D.to_csv('pricepredict.csv')


# In[38]:


plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.xlabel('area(sqr ft)')
plt.ylabel('price(us$)')


# Linear Rgression with Multiple Varaibles
# Multivariate Regression

# In[39]:


from sklearn import linear_model


# In[40]:


df=pd.read_csv('test12.csv')


# In[41]:


df


# In[42]:


import math
median_bedrooms=math.floor(df.bedrooms.median()) # To find the median of bedrooms
median_bedrooms


# In[43]:


df.bedrooms=df.bedrooms.fillna(median_bedrooms) # To fill the NaN values
df


# In[44]:


# given this home prices find out the price of a home that has,
# 3000 area, 3 bedrooms,40 years
#2500 area,4 bedrooms, 5 years


# In[45]:


reg=linear_model.LinearRegression()      
reg.fit(df[['area','bedrooms','age']],df.price) # linear regression with multiple independet variables


# In[46]:


reg.coef_


# In[47]:


reg.intercept_


# In[48]:


reg.predict([[3000,4,15]])


# In[49]:


# maths beyond the value
13.555*3000+ -2449.5*4 + -673.5*15 +38255.49999


# In[50]:


# 3000 area, 3 bedrooms,40 years
reg.predict([[3000,3,40]])


# In[51]:


#2500 area,4 bedrooms, 5 years
reg.predict([[2500,4,5]])


# In[52]:


df = pd.read_excel (r'C:\Users\director\Downloads\salary.xls')


# In[53]:


print(df)


# In[54]:


from word2number import w2n
df.experience=df.experience.fillna("zero")


# In[55]:


df


# In[56]:


df.experience = df.experience.apply(w2n.word_to_num)
df


# In[57]:


test_score=math.floor(df['test_score(10)'].mean())
test_score


# In[58]:


df['test_score(10)']=df['test_score(10)'].fillna(test_score) # To fill the NaN values
df


# In[59]:


reg=linear_model.LinearRegression()      
reg.fit(df[['experience','test_score(10)','interview_score(10)']],df['salary($)']) # linear regression with multiple independet variables


# In[60]:


reg.coef_


# In[61]:


reg.intercept_


# In[62]:


reg.predict([[10,9,9]])


# Logistic Regression Binary Classification or Simple Classification(Y/N,1/0 etc.)
# 
# 
# Note: Logistic Regression uses the Sigmoid(z) or Logit function converts input into range of 0 or 1
# sigmoid(z) =1/1+e**-z

# In[63]:


df=pd.read_excel(r'C:\Users\director\Downloads\Logistic.xls')


# In[64]:


df.head()


# In[65]:


plt.scatter(df.age,df.brought_insurance, marker='+',color='red')


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


x_train, x_test, y_train, y_test =train_test_split(df[['age']],df.brought_insurance,test_size=0.3)

x_test


# In[68]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[69]:


model.fit(x_train,y_train)


# In[70]:


model.predict(x_test)


# In[71]:


model.score(x_test,y_test) # model.score used to know the accuracy of the model
# in this case our model is perfect because score is 1


# In[72]:


model.predict_proba(x_test) # this is used to know the probability of each indivdual to take the insurance


# Logistic Regression for Multiclassification

# In[73]:


from sklearn.datasets import load_digits


# In[74]:


digits=load_digits()


# In[75]:


dir(digits)


# In[76]:


digits.data[0]


# In[77]:


plt.gray()


# In[78]:


for i in range(5): # to display the sample of 5 images from the dataset
    plt.matshow(digits.images[i])


# In[79]:


digits.target[0:5]


# In[80]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)
len(x_train)


# In[81]:


len(x_test)


# In[82]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[83]:


model.fit(x_train,y_train)


# In[84]:


model.score(x_train,y_train)


# In[85]:


model.predict([digits.data[67]])


# In[86]:


model.predict(digits.data[1:5])


# In[87]:


y_predicted=model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm


# In[88]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# Types of Classification
# 
# Decision Tree: Graphical representation of all the possible solutions to a decision
#                Decisions are based on some condiditions
#                Decsions made can be easily be explained
#                
# Random Forest: Builds multiple decsion trees and merges them together
#                More accurate and stable prediction
#                Random decsion forests correct for decision trees's habit of overfitting to their training set
#                trained with the 'bagging' method
# 
# Naive Bayes  : Classification technique based on Bayes' Theorm
#                Assumes that the presence of a particluar feature in a class is unrelated to the presence of any other feature
#                
# KNN  : Stores all the availble cases and classifies new cases based on a similariy measure
#                The K in KNN algorithm is the nearest neighbours we wish to take vote from.

# Decsion Tree Terminology
# Leaf Node -Node cannot be further segregated into further nodes
# Spliting  -Dividing the root node into diffrent parts on the basis of some condtions
# Root Node -It represents the entire population or sample and this further gets divided into two or more homogenous sets
# Bandch/Sub Tree- Formed by splitting the tree/node
# Pruning- Opposite of splitting. Basically removing unwanted branches from the tree
# Parent/Child Node- root is the parent node and all other nodes are child nods
# 
#     CART Algorithm
# Gini index: the measure of impurity (or purity) used in building decision tree in CART is Gini index
# 
# Information Gain: The information gain is the decrease in entropy after a dataset is split on the basis of attribute.
#                   Constructing a decsion tree is all about finding attribute that returns the highest information gain
#                   
# Reduction in Varuance: It is used for regression problems. the split with lower variance is selected as the criteria to split
#                        the population.
#                        
# Chi Square: It is used to find the statistical significance between the differences between sub-nodes and parent node.
# 
#     Entropy: Is just a metric which measures the impurity 
#     impurity: Defines(dgree of) randomness in the data
#     if number of Yes = number of No (Entropy =1) .i.e P(s)=0.5
#     if it contains all Yes or all No (Entorpy =0) .i.e P(s)= 1 or 0
# 

# In[ ]:





# In[ ]:





# In[92]:





# In[95]:





# In[98]:





# In[99]:





# In[103]:





# In[104]:





# In[105]:





# In[ ]:





# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:




