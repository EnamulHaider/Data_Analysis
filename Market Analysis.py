#!/usr/bin/env python
# coding: utf-8

# In[43]:


#In this portfolio project we will be looking at data from the stock market, particularly some technology stocks
# for Division

from __future__ import division

import pandas as pd
from pandas import Series,DataFrame
import numpy as np

#For Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#For reading stockmarket data from yahoo
import pandas_datareader as pdr
from pandas_datareader import data,wb
#For time stamp
from datetime import datetime

tech_list = ['AAPL','GOOG','MSFT','AMZN']
end = datetime.now()
start = datetime(end.year-1,end.month,end.day)


for stock in tech_list:
    globals()[stock] = pdr.DataReader(stock,'yahoo',start,end)

AAPL


# In[44]:


AAPL.info()


# In[45]:


AAPL['Adj Close'].plot(legend=True,figsize=(10,4))


# In[46]:


AAPL['Volume'].plot(legend=True,figsize=(10,4))


# In[47]:


#Luckily pandas has a build-in rolling mean calculator 
# Let's go ahead and plot out several moving averages
ma_day = [10,20,30]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    AAPL[column_name]=AAPL['Adj Close'].rolling(ma).mean()


# In[48]:


AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 30 days']].plot(subplots=False,figsize=(10,4))


# In[49]:


#Plotting % Change to see how much I have on daily return

AAPL['Daily Return']=AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(10,4),legend=True,linestyle='--',marker='o')


# In[50]:


# Using some of the capabilities on Seaborn like displot to place two plot on each other

sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# In[51]:


#Using Pandas build in histogram 

AAPL['Daily Return'].hist(bins=100)


# In[52]:


#Created a new data frame to read data from web using pandas

closing_df= pdr.DataReader(tech_list,'yahoo',start,end)['Adj Close']


# In[53]:


closing_df.head()


# In[54]:


#Get the daily returns

tech_rets= closing_df.pct_change()


# In[55]:


tech_rets.head()


# In[56]:


#Compare the daily returns between the two stocks to see how they are corelated 
#Compaing google to google

sns.jointplot('GOOG','GOOG',tech_rets,kind='scatter',color='seagreen')


# In[57]:


#Lets compare to two different stocks
sns.jointplot('GOOG','MSFT',tech_rets,kind="scatter")


# In[58]:


#Checking Coorelation using Seaborn , Pair Plots 

from IPython.display import SVG
tech_rets.head()


# In[59]:


#Lets drop na values and create a pairplot
sns.pairplot(tech_rets.dropna())


# In[60]:


#Let's break it down and reevaluate
returns_fig= sns.PairGrid(tech_rets.dropna())
#Lets have scatter plot
returns_fig.map_upper(plt.scatter,color='purple')
# Lets call KDE plot on the lower
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
#On the diagnal lets call the histogram 
returns_fig.map_diag(plt.hist,bins=30)


# In[61]:


#Let's break it down and reevaluate for closing prices
returns_fig= sns.PairGrid(closing_df)
#Lets have scatter plot
returns_fig.map_upper(plt.scatter,color='purple')
# Lets call KDE plot on the lower
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
#On the diagnal lets call the histogram 
returns_fig.map_diag(plt.hist,bins=30)


# In[62]:


#To check out the actual value of co-relation , we do the actual heatmap with corelation functiom
sns.heatmap(tech_rets.dropna().corr(),annot=True)


# In[63]:


#Lets check on closing date
sns.heatmap(closing_df.corr(),annot=True)


# In[64]:


# Let's clean the version of the tech_rets DataFrame
rets = tech_rets.dropna()


# In[65]:


# Let's start by defining a new DataFrame as a clenaed version of the orignal tech_rets DataFrame
area = np.pi*20
plt.scatter(rets.mean(),rets.std(),alpha=0.5,s=area)

#set the x and y limits for the plot
plt.ylim([0.01,0.025])
plt.xlim([-0.003,0.004])
#Set the plot axis tiles
plt.xlabel('Expected Return')
plt.ylabel('Risk')

# Label the scatter plots, for more info on how this is done
# http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


# In[66]:


#The use of dropna() to remove NaN
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# In[67]:


#0.05 empirical quantity of daily returns
rets['AAPL'].quantile(0.05)


# In[68]:


#Set up horizontal axis
days= 365

#now the delta
dt=1/days

#Now let's grab mu(drift) from expected return date 
mu=rets.mean()['GOOG']

#Now lets grab the volatility of stock from the std() of the average return 
sigma= rets.std()['GOOG']



# In[69]:


def stock_monte_carlo(start_price,days,mu,sigma):
    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''
    
    # Define a price array
    price = np.zeros(days)
    price[0] = start_price
    # Schok and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Run price array for number of days
    for x in xrange(1,days):
        
        # Calculate Schock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[x] = mu * dt
        # Calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price


# In[70]:


# Get start price from GOOG.head()
start_price = 569.85

for run in xrange(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Google')


# In[71]:


#Set  a large number of runs
runs= 10000
#Create an empty matrix to hold the end price data
simulations =np.zeros(runs)
#Set the print options of numpy to only display 0-5 points from an array to surpress output
np.set_printoptions(threshold=5)

for run in xrange(runs):
    #Set  the simulation data point as the last stock price for that run
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1];


# In[73]:


#Now we'll define q as a 1% empirical quantity 
q=np.percentile(simulations,1)

#Now lets plot the distribution of the end price
plt.hist(simulations,bins=200)

#Starting price
plt.figtext(0.6,0.8,s="Start price: $%.2f" %start_price)

#Mean ending price
plt.figtext(0.6,0.8,s="Mean final price: $%.2f" % simulations.mean())

#Variance of price (within 99% confidence interval)
plt.figtext(0.6,0.8,s="var(0.99): $%.2f" %(start_price - q,))

#Display 1% quantile
plt.figtext(0.15,0.6,"q(0.99): $%.2f" % q)

#Plot a line at the 1% quantile result 
plt.axvline(x=q, linewidth=4,color='r')

#Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');


# In[ ]:




