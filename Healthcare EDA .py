#!/usr/bin/env python
# coding: utf-8

# # Health Care Exploratory Data Analysis

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv(r"C:\Users\Vishal Kumar\Downloads\healthcare_messy.csv")


# In[5]:


df.head(10)


# In[8]:


df.shape


# In[9]:


df.info()


# In[14]:


df.isnull().sum()


# In[12]:


df.duplicated().sum


# Data Processing

# In[15]:


df['age']=df['age'].fillna(df['age'].median())
df['age']=df['age'].astype(int)
df['gender']=df['gender'].fillna('unknown')
df['admission_date']=pd.to_datetime(df['admission_date'],errors='coerce')
df['admission_year']=df['admission_date'].dt.year
df['disease']=df['disease'].str.strip().str.lower()
df['disease']=df['disease'].fillna('unknown')


# In[16]:


Q1 = df['treatment_cost'].quantile(0.25)
Q3=  df['treatment_cost'].quantile(0.75)
IQR = Q3-Q1 

upper_limit = Q3 + 1.5* IQR


# In[17]:


df['treatment_cost']=np.where(df['treatment_cost'] > upper_limit,upper_limit,df['treatment_cost'])


# In[18]:


df['hospital_type']=df['hospital_type'].str.strip().str.lower()
df['hospital_type']=df['hospital_type'].fillna('unknown')
df['insurance']=df['insurance'].str.strip().str.lower()
df['insurance']=df['insurance'].fillna('no')


# In[19]:


df['lenght_of_stay']=df['length_of_stay'].fillna(df['length_of_stay'].median())


# In[20]:


df=df[df['length_of_stay']>=0]
df['length_of_stay']=df['length_of_stay'].astype(int)


# In[21]:


df['outcome']=df['outcome'].str.strip().str.lower()
df['outcome']=df['outcome'].fillna('unknown')


# In[22]:


df.info()


#                                                           # Data Plotting

# AGE DISTRIBUTION

# In[41]:


plt.figure(figsize=(5,4))
plt.hist(df['age'],bins=50)
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.title('Age Distribution of Patients')
plt.tight_layout()
plt.savefig('age distribution.png',dpi=200)
plt.show()


# Gender Distribution
# 

# In[40]:


gender_counts = df['gender'].value_counts()
plt.figure(figsize=(5,4))
plt.bar(gender_counts.index,gender_counts.values)

plt.xlabel('Gender')
plt.ylabel('Number of Patients')
plt.title('Gender Distributions')
plt.tight_layout()
plt.savefig('Gender distribution.png', dpi=300)
plt.show()


# In[26]:


df['gender'].value_counts()


# Disease Distribution

# In[39]:


plt.figure(figsize=(5,4))
df['disease'].value_counts().head(10).plot(kind='bar')
plt.xlabel('Disease')
plt.ylabel('Number of Patients')
plt.title('Top 10 Diseases')
plt.tight_layout()
plt.savefig('Disease distribution.png', dpi=300)
plt.show()


# In[28]:


df['disease'].value_counts()


# Treatment Cost Distribution

# In[38]:


plt.figure(figsize=(5,4))
plt.boxplot(df['treatment_cost'])
plt.ylabel('Treatment Cost')
plt.title('Distribution of Treatment Cost')
plt.tight_layout()
plt.savefig('Distribution of treatment cost.png', dpi=300)
plt.show()


# In[30]:


df['treatment_cost'].value_counts().values


# 
# Insurance VS Average Treatment Cost

# In[45]:


plt.figure(figsize=(5,4))
df.groupby('insurance')['treatment_cost'].mean().plot(kind='bar')
plt.xlabel('Insurance')
plt.ylabel('Average Treatment Cost')
plt.title('Average Treatment Cost by Insurance Status')
plt.tight_layout()
plt.savefig('insurance vs treatment cost.png', dpi=300)
plt.show()


# In[43]:


df[['insurance','treatment_cost']].value_counts()


# In[35]:


pd.pivot_table(df,index='treatment_cost',columns='insurance',aggfunc='size',
fill_value=0)
df[df['treatment_cost']>=0].head(20)


# Disease vs Treatment Cost

# In[36]:


plt.figure(figsize=(5,4))
df.boxplot(column='treatment_cost', by='disease', rot=45)
plt.xlabel('Disease')
plt.ylabel('Treatment Cost')
plt.title('Treatment Cost by Disease')
plt.tight_layout()
plt.savefig('cost_by_disease.png', dpi=300)
plt.show()


#     Age vs treatment Cost

# In[9]:


plt.figure(figsize=(5,4))
plt.scatter(df['age'], df['treatment_cost'])
plt.xlabel('Age')
plt.ylabel('Treatment Cost')
plt.title('Age vs Treatment Cost')
plt.tight_layout()
plt.savefig('age_vs_cost.png', dpi=300)
plt.show()


#     ALL Charts Plottinng

# In[10]:


fig, axes = plt.subplots(2, 4, figsize=(22, 10))

# 1️⃣ Age Distribution
axes[0, 0].hist(df['age'], bins=20)
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Patients')

# Gender Distribution
gender_counts = df['gender'].value_counts()
axes[0, 1].bar(gender_counts.index, gender_counts.values)
axes[0, 1].set_title('Gender Distribution')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Count')

# Disease Distribution (Top 10)
disease_counts = df['disease'].value_counts().head(10)
axes[0, 2].bar(disease_counts.index, disease_counts.values)
axes[0, 2].set_title('Top Diseases')
axes[0, 2].set_xlabel('Disease')
axes[0, 2].set_ylabel('Patients')
axes[0, 2].tick_params(axis='x', rotation=45)

# Treatment Cost Distribution
axes[0, 3].boxplot(df['treatment_cost'])
axes[0, 3].set_title('Treatment Cost')
axes[0, 3].set_ylabel('Cost')

# Insurance vs Avg Treatment Cost
insurance_avg = df.groupby('insurance')['treatment_cost'].mean()
axes[1, 1].bar(insurance_avg.index, insurance_avg.values)
axes[1, 1].set_title('Avg Cost by Insurance')
axes[1, 1].set_xlabel('Insurance')
axes[1, 1].set_ylabel('Avg Cost')

# Disease vs Treatment Cost
df.boxplot(column='treatment_cost', by='disease', ax=axes[1, 0], rot=45)
axes[1, 0].set_title('Cost by Disease')
axes[1, 0].set_xlabel('Disease')
axes[1, 0].set_ylabel('Cost')

# Age vs Treatment Cost
axes[1, 2].scatter(df['age'], df['treatment_cost'])
axes[1, 2].set_title('Age vs Treatment Cost')
axes[1, 2].set_xlabel('Age')
axes[1, 2].set_ylabel('Cost')

# Empty subplot (clean look)
axes[1, 3].axis('off')

# Remove automatic pandas title
plt.suptitle('All Plots')

# Adjust spacing
plt.tight_layout()

# Save single image
plt.savefig('all_eda_charts_one_figure.png', dpi=300)

plt.show()


# In[ ]:




