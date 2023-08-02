#!/usr/bin/env python
# coding: utf-8

# # Exploring dataset of tumor drug testing in mice

# In[1]:


import os
import scipy
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH_DATA = os.path.join(os.getcwd(), 'dataset', 'Mouse_metadata.csv')
CSV_PATH_RESULT = os.path.join(os.getcwd(), 'dataset', 'Study_results.csv')

#dataloading
data = pd.read_csv(CSV_PATH_DATA)
result = pd.read_csv(CSV_PATH_RESULT)

#data has 248 rows x 5
#result has 1893 rows x 4 -> remove dups : 1880 rows


# In[2]:


print(data.shape)
print(data.columns)
data.head(3)


# In[3]:


print(result.shape)
print(result.columns)
result.head(3)


# #### Removing the mouse with duplicated timepoints and removing any data associated with that mouse

# In[4]:


dups = result[result.duplicated(["Mouse ID","Timepoint"])] #dfObj[dfObj.duplicated(['Age', 'City'])]
dups
# Mouse ID g989 has duplicated values


# In[5]:


result = result.drop_duplicates(subset=["Mouse ID","Timepoint"])


# In[6]:


result


# In[7]:


#Making sure to remove all data associated with mouse g989

result = result[result["Mouse ID"] != "g989"]
data = data[data["Mouse ID"] != "g989"]


# In[8]:


df3 = pd.merge(data, result, on=['Mouse ID'])
print(df3.shape)
df3.sample(6)


# #### Exploring the merged dataframes, df3

# In[9]:


# Number of drugs tested
df3["Drug Regimen"].nunique()


# In[10]:


distribution = df3.groupby('Drug Regimen')['Tumor Volume (mm3)'].agg(['mean','median','var','std','sem'])
distribution


# In[11]:


# Pivot table of Mouse count 

df4 = pd.pivot_table(df3, values='Mouse ID', index = 'Timepoint', columns = 'Drug Regimen',aggfunc='count')
df4

# The study started with a balanced numbers of mice across all drugs tested.
# Capomulin and Ramicane have the most number of mice at the end of the study.


# In[12]:


sns.set(rc = {'figure.figsize':(15, 10)},font_scale=1.5)
df4.plot(xticks=df4.index, ylabel='Drug Regimen')

# Apart from Capomulin and Ramicane, all other drugs have less than 15 mice at the end of the study.


# #### Generating a bar plot using both Pandas's DataFrame.plot() and Matplotlib's pyplot that shows the number of total mice for each treatment regimen throughout the course of the study.

# In[13]:


# Using pandas

df3.groupby(by='Drug Regimen')['Mouse ID'].count().plot.bar()


# In[14]:


# Using pyplot

mice_count = df3.groupby(by='Drug Regimen')['Mouse ID'].count().reset_index() 
# a series,sorted by name of Drug # with reset_index() it will change series to df
mice_count


# In[15]:


fig, ax = plt.subplots(figsize=(10,5))

ax.bar(mice_count['Drug Regimen'], mice_count['Mouse ID'])
ax.set_xticks(mice_count['Drug Regimen'])
ax.set_xticklabels(mice_count['Drug Regimen'],rotation=90)
ax.set_ylabel('Number of Mice')
ax.set_title('Number of Mice for Each Drug')

plt.show()


# #### Generate a pie plot that shows the distribution of female or male mice in the study.

# In[16]:


# Using pandas
s = df3['Sex'].value_counts()

plt.title("Female vs. Male in Study")
s.plot.pie(autopct='%1.1f%%',figsize=(5, 5))

# Using pyplot
sex = df3['Sex'].value_counts()
plt.figure(figsize=(5,5))
plt.pie(sex, labels = sex.index, autopct='%1.0f%%') # < This will show the values with %
plt.title('Percentages of Male vs Female Sample')
plt.show()


# #### Finding the final tumor volume of each mouse across four of the most promising treatment regimens: Capomulin, Ramicane, Infubinol, and Ceftamin. 

# In[17]:


# Making df for each drug
df_capo = df3[df3['Drug Regimen'] == 'Capomulin']
df_rami = df3[df3['Drug Regimen'] == 'Ramicane']
df_infu = df3[df3['Drug Regimen'] == 'Infubinol']
df_cef = df3[df3['Drug Regimen'] == 'Ceftamin']


# In[18]:


# Final tumor volume for each drug, is at the last Timepoint

#Capomulin
final_capo = df_capo.groupby('Mouse ID')['Timepoint'].max() # A series
df_finalcapo = pd.DataFrame(final_capo) # change series to df
capomerge = pd.merge(df_capo, df_finalcapo, on = ('Mouse ID','Timepoint'), how='right')
#capomerge.head()# 2 cols: mouse ID & max timepoint (25,1)

#Ramicane
final_rami = df_rami.groupby('Mouse ID')['Timepoint'].max() # A series
df_finalrami = pd.DataFrame(final_rami)
ramimerge = pd.merge(df_rami, df_finalrami, on = ('Mouse ID','Timepoint'), how='right')
#ramimerge.head()# 2 cols: mouse ID & max timepoint (25,1)

#Infubinol
final_infu = df_infu.groupby('Mouse ID')['Timepoint'].max() # A series
df_finalinfu = pd.DataFrame(final_infu)
infumerge = pd.merge(df_infu, df_finalinfu, on = ('Mouse ID','Timepoint'), how='right')
#infumerge.head()# 2 cols: mouse ID & max timepoint (25,1)

# Ceftamin
final_cef = df_cef.groupby('Mouse ID')['Timepoint'].max() # A series
df_finalcef = pd.DataFrame(final_cef)
cefmerge = pd.merge(df_cef, df_finalcef, on = ('Mouse ID','Timepoint'), how='right')
#cefmerge.head()# 2 cols: mouse ID & max timepoint (25,1)


# In[19]:


# Finding quartile
# quartiles:  0.25, 0.5, 0.75
# IQR: 0.75 - 0.25
# Outliers: 0.75 + 1.5 (Upper), 0.25 - 1.5 (lower)

# Capomulin
capo_q3 = np.quantile(capomerge['Tumor Volume (mm3)'], 0.75)
capo_q1 = np.quantile(capomerge['Tumor Volume (mm3)'], 0.25)
capo_q2 = np.quantile(capomerge['Tumor Volume (mm3)'], 0.5)
print(f'Capomulin q1 is {round(capo_q1,2)}, q2 is {round(capo_q2,2)}, and q3 is {round(capo_q3,2)}.')

capo_iqr = round(capo_q3 - capo_q1,2)
print('Capomulin IQR is',capo_iqr)
print()

# Ramicane
rami_q3 = np.quantile(ramimerge['Tumor Volume (mm3)'], 0.75)
rami_q1 = np.quantile(ramimerge['Tumor Volume (mm3)'], 0.25)
rami_q2 = np.quantile(ramimerge['Tumor Volume (mm3)'], 0.5)
print(f'Ramicane q1 is {round(rami_q1,2)}, q2 is {round(rami_q2,2)}, and q3 is {round(rami_q3,2)}.')

rami_iqr = round(rami_q3 - rami_q1,2)
print('Ramicane IQR is', rami_iqr)
print()

# Infubiol
infu_q3 = np.quantile(infumerge['Tumor Volume (mm3)'], 0.75)
infu_q1 = np.quantile(infumerge['Tumor Volume (mm3)'], 0.25)
infu_q2 = np.quantile(infumerge['Tumor Volume (mm3)'], 0.5)
print(f'Infubinol q1 is {round(infu_q1,2)}, q2 is {round(infu_q2,2)}, and q3 is {round(infu_q3,2)}.')

infu_iqr = infu_q3 - infu_q1
print('Infubiol IQR is',round(infu_iqr,2))
print()

# Ceftamin
cef_q3 = np.quantile(cefmerge['Tumor Volume (mm3)'], 0.75)
cef_q1 = np.quantile(cefmerge['Tumor Volume (mm3)'], 0.25)
cef_q2 = np.quantile(cefmerge['Tumor Volume (mm3)'], 0.5)
print(f'Ceftamin q1 is {round(cef_q1,2)}, q2 is {round(cef_q2,2)}, and q3 is {round(cef_q3,2)}.')

cef_iqr = cef_q3 - cef_q1
print('Ramicane IQR is',round(cef_iqr,2))


# In[20]:


# Finding Outliers

# Capomulin
lower_capo = capo_q1 - 1.5 * capo_iqr
upper_capo = capo_q1 + 1.5 * capo_iqr

# Ramicane
lower_rami = rami_q1 - 1.5 * rami_iqr
upper_rami = rami_q1 + 1.5 * rami_iqr

# Infubiol
lower_infu = infu_q1 - 1.5 * infu_iqr
upper_infu = infu_q1 + 1.5 * infu_iqr

# Ceftamin
lower_cef = cef_q1 - 1.5 * cef_iqr
upper_cef = cef_q1 + 1.5 * cef_iqr

print(f'Capomulin lower outlier is {round(lower_capo,2)},and its upper outlier is {round(upper_capo,2)}.\n')
print(f'Ramicane lower outlier is {round(lower_rami,2)},and its upper outlier is {round(upper_rami,2)}.\n')
print(f'Infubinol lower outlier is {round(lower_infu,2)},and its upper outlier is {round(upper_infu,2)}.\n')
print(f'Ceftamin lower outlier is {round(lower_cef,2)},and its upper outlier is {round(upper_cef,2)}.')


# In[21]:


# Merge ALL df

df_all = pd.concat([capomerge,ramimerge,infumerge,cefmerge], ignore_index=True)
df_all.sample()
df_all.shape #(100,8)


# #### Generating a box and whisker plot of the final tumor volume for all four treatment regimens to visualize potential outliers

# In[22]:


fig = plt.figure(figsize=(10,10))
sns.boxplot( x = 'Drug Regimen',y = 'Tumor Volume (mm3)', data = df_all)
plt.show()


# In[23]:


b128 = df_capo[df_capo['Mouse ID'] == 'b128']
b128


# #### Selecting a mouse that was treated with Capomulin and generate a line plot of tumor volume vs. time point for that mouse.

# In[24]:


tp = b128['Timepoint']
vol = b128['Tumor Volume (mm3)']

plt.plot(tp,vol)
plt.xlabel('Timepoint')
plt.ylabel('Tumor Volume (mm3)')
plt.title('Tumor volume vs. Timepoint')
plt.show()


# #### Generate a scatter plot of mouse weight versus average tumor volume for the Capomulin treatment regimen.

# In[25]:


avg_vol = df_capo.groupby(['Mouse ID']).mean()

# plt.scatter(avg_vol['Weight (g)'],avg_vol['Tumor Volume (mm3)'])

plt.plot(avg_vol['Weight (g)'],avg_vol['Tumor Volume (mm3)'],'go')
plt.show()


# #### Finding correlation coefficient and linear regression model between mouse weight and average tumor volume for the Capomulin treatment. 
# 
# #### Plotting the linear regression model on top of the previous scatter plot.

# In[26]:


plt.title('Mouse weight and average tumor volume')
reg = sns.regplot(x='Weight (g)', y='Tumor Volume (mm3)', data=avg_vol)

corr = avg_vol['Tumor Volume (mm3)'].corr(avg_vol['Weight (g)'])
print('The correlation coefficient is {}'.format(round(corr,2)))

slope, intercept, r, reg, sterr = scipy.stats.linregress(x=reg.get_lines()[0].get_xdata(),
                                                       y=reg.get_lines()[0].get_ydata())

#add regression equation to plot
plt.text(16,46,'y = ' + str(round(intercept,3)) + ' + ' + str(round(slope,3)) + 'x',
        fontsize = 12,bbox = dict(facecolor = 'red', alpha = 0.5))


# ### Observations and Inferences:
# 
# 1. Balanced gender population of mouse
# 2. Ramicane and Capomulane are more effective  to treat tumor
# 3. There's a high correlation between mouse weight and average tumor volume.
# 4. Interestingly, tumor volume increases after about timepoint 33, after a sharp reduction from around timepoint 24
