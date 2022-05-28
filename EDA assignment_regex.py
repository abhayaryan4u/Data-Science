#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import seaborn as sns


# In[3]:


heart=pd.read_csv('heart.csv')


# In[4]:


heart.head()


# In[5]:


sns.distplot(heart['age'])
plt.show()


# In[6]:


heart.shape


# In[7]:


heart.head()


# In[8]:


heart.describe()


# In[9]:


heart['sex'].value_counts()


# In[10]:


heart['sex'].value_counts().keys()


# In[11]:


heart['sex'].value_counts().values


# In[12]:


plt.bar(list(heart['sex'].value_counts.keys()),list(heart['sex'].value_counts()))


# In[13]:


plt.bar(list(heart['sex'].value_counts().keys()),list(heart['sex'].value_counts()))


# In[14]:


plt.bar(list(("male","female")),list(heart['sex'].value_counts()),color=["blue","pink"])


# In[15]:


heart.head()


# In[16]:


heart['cp'].value_counts()


# In[18]:


plt.bar(list(("level-0","level-1","level-2","level-3")),list(heart['cp'].value_counts()),color=["green","yellow","orange","red"])


# In[19]:


heart.head()


# In[20]:


sns.distplot(heart['chol'])
plt.show()


# In[21]:


heart['target'].value_counts()


# In[23]:


plt.bar(list(("not-safe","safe")),list(heart['target'].value_counts()),color=["red","green"])


# In[24]:


heart.head()


# In[27]:


x=heart[['age']]
y=heart[['target']]


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[30]:


from sklearn.naive_bayes import MultinomialNB


# In[31]:


mnb=MultinomialNB()


# In[32]:


mnb.fit(x_train,y_train)


# In[33]:


y_pred=mnb.predict(x_test)


# In[34]:


y_test.head(),y_pred[0:5]


# In[35]:


from sklearn.metrics import confusion_matrix


# In[37]:


confusion_matrix(y_test,y_pred)


# In[38]:


(158)/(158+150)


# In[39]:


#accuracy is low so let's create another model with more than 1 predictor variables


# In[40]:


heart.head()


# In[41]:


x= heart[['age','trestbps','chol']]


# In[42]:


x.head()


# In[43]:


y=heart[['target']]


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)


# In[46]:


from sklearn.naive_bayes import GaussianNB


# In[48]:


gnb= GaussianNB()


# In[50]:


gnb.fit(x_train,y_train)


# In[51]:


y_pred=gnb.predict(x_test)


# In[52]:


from sklearn.metrics import confusion_matrix


# In[53]:


confusion_matrix(y_test,y_pred)


# In[54]:


(120+146)/(146+60+84+120)


# In[ ]:


#accuracy is increased as compared to previous one

