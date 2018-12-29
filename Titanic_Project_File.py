import numpy as np
import pandas as pd
from pandas import Series, DataFrame
titanic_df= pd.read_csv("C:/data/train.csv")
# Age/Sex Distribution 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.catplot( 'Sex', data=titanic_df,kind="count")
sns.catplot('Sex',data=titanic_df, kind='count',hue='Pclass')
sns.catplot('Pclass',data=titanic_df, kind='count', hue='Sex')
def male_female_child(passenger):
    age,sex=passenger
    if age < 16 :
        return 'child'
    else :
        return sex
titanic_df['person']= titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
titanic_df[0:10]
sns.catplot('Pclass',data=titanic_df,kind='count',hue='person')
titanic_df['Age'].hist(bins=70)
titanic_df['Age'].mean()
titanic_df['person'].value_counts()

fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

#What deck where passengers on based on their class
deck = titanic_df['Cabin'].dropna()


levels =[]
for level in deck:
    levels.append(level[0])
    
cabin_df =DataFrame(levels)
cabin_df.columns =['Cabin']
sns.catplot('Cabin', data=cabin_df, kind='count',palette='winter_d')

cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.catplot('Cabin',data=cabin_df,kind='count',palette ='summer')


#Where did the passengers came from 

sns.catplot('Embarked', data=titanic_df,kind='count',hue='Pclass')

#Who is alone and who is with family?

titanic_df['Alone'].loc[titanic_df['Alone']>0] ='With Family'
titanic_df['Alone'].loc[titanic_df['Alone']==0]='Alone'

sns.catplot('Alone', data=titanic_df,kind='count', palette='Blues')

titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})

sns.catplot('Survivor', data=titanic_df,kind='count',palette='Set1')

sns.lineplot(x='Pclass',y='Survived',hue='person',data=titanic_df)

sns.lmplot('Age','Survived',data=titanic_df)

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter')

generation =[10,20,40,60,80]

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generation)

sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generation)


