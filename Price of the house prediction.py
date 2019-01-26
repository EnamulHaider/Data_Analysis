#!/usr/bin/env python
# coding: utf-8

# In[59]:


#set up library
import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[60]:


import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')


# In[61]:


from sklearn.datasets import load_boston


# In[62]:


#imported the boston dataset
boston =load_boston()


# In[63]:


print boston.DESCR


# In[64]:


plt.hist(boston.target,bins=50)

plt.xlabel('Prices in $1000')
plt.ylabel('Number of Houses')


# In[65]:


plt.scatter(boston.data[:,5],boston.target)
plt.ylabel('Price in $1000s')
plt.xlabel('Number of rooms')


# In[87]:


#Store the data in dataframe
boston_df = DataFrame(boston.data)
boston_df.columns= boston.feature_names


# In[88]:


boston_df.head()


# In[89]:


boston_df['Price']= boston.target


# In[90]:


boston_df.head()


# In[91]:


sns.lmplot('RM','Price',data=boston_df)


# In[105]:


#Set up X as median room 
X = boston_df.RM


# In[106]:


#to reform the values
X= np.vstack(boston_df.RM)
#How many values and attributes
X.shape

#to reform the values
X= np.vstack(boston_df.RM)


# In[107]:


#set up target price as y
##y = mx + b which converts to y = Ap where A = [x+1] and p=[m 
#############################################################b]
##It is equal to  y= mx+b
Y= boston_df.Price


# In[108]:


X= boston_df.RM
X = np.vstack([X, np.ones(len(X))]).T
#linear fit value m and b and we need only one index value
m,b = np.linalg.lstsq(X,Y,rcond=-1)[0]


# In[109]:


#create a scatter plot
plt.plot(boston_df.RM,boston_df.Price,'o')
#plot best fit line
x= boston_df.RM
plt.plot(x,m*x+ b,'r',label='Best Fit Line')


# In[113]:


#Calculate the root mean square error : 95% of time the values will be within 2 times root mean square errror
result = np.linalg.lstsq(X,Y)
error_total = result[1]
rmse = np.sqrt(error_total/len(X))
print 'The root mean square error was %.2f' %rmse


# In[114]:


#linear regression library
import sklearn
from sklearn.linear_model import LinearRegression 


# In[115]:


#We'll create linear regression object 
lreg = LinearRegression()


# In[117]:


#Split boston data columns
#Everything in dataframe minus Price
X_multi = boston_df.drop('Price',1)
#Only the price 
Y_target = boston_df.Price


# In[118]:


#we made an equation of line with 13 coeffiecient 
lreg.fit(X_multi,Y_target)


# In[120]:



print 'The estimate interceptcoefficient is %.2f' %lreg.intercept_
print 'The number of coefficients used was %d' %len(lreg.coef_)


# In[122]:


coeff_df = DataFrame(boston_df.columns)
coeff_df.columns =['Feature']
coeff_df['Coefficient Estimate'] = Series(lreg.coef_)
coeff_df


# In[123]:


#The highest correlated feature is number of rooms


# In[128]:


#predict our prices and residual plots
#split the data set into training and validation models
#Cross validation library does train and analysis split
X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(X,boston_df.Price)


# In[130]:


#majority sits in training sets and rest in testing sets
print X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


# In[131]:


#Predict price 
#House Price
lreg=LinearRegression()
lreg.fit(X_train,Y_train)


# In[132]:


pred_train=lreg.predict(X_train)
pred_test = lreg.predict(X_test)


# In[134]:


print "Fit a model X_train, and calculate the MSE with Y_train: %.2f" %np.mean((Y_train-pred_train)**2)
print "Fit the model with my training data set and calculate the MSE with my X_test and Y Test: %.2f" % np.mean((Y_test-pred_test)**2)


# In[140]:


#With the top result we are not superclose or superoff 
#Residual value E = Observed value - Predicted value
#Residual plot to visualize the error 
#RESIDUAL PLOT
train = plt.scatter(pred_train,(pred_train-Y_train),c='b',alpha= 0.5)

test = plt.scatter(pred_test,(pred_test-Y_test),c='r',alpha=0.5)

plt.hlines(y=0,xmin=-10,xmax=40)

plt.legend((train,test),('Training','Test'),loc='lower left')

plt.title('Residual Plot')


# In[ ]:


###The line is between the spread, so linear regression was the best choice here.

