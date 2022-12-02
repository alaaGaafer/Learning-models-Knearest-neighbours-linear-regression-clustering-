#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sales=pd.read_csv(r'C:\Users\alaau\OneDrive\Desktop\Assignment Three(1)\sales.csv')   
fuel=pd.read_csv(r'C:\Users\alaau\OneDrive\Desktop\Assignment Three(1)\fuel.csv')   
weather=pd.read_csv(r'C:\Users\alaau\OneDrive\Desktop\Assignment Three(1)\weather.csv')   
print(sales.describe())
print("-----------------------------Dataframe info----------------------------------")
print(sales.info())
print("--------------------------NULL sum-------------------------------------")
print(sales.isnull().sum())
sales.head(10)
# No nulls in sales dataframe
# weekly sales have negative valuse as seen from min row
# there is 45 store,99 category.
# date is an object which may cause a problem


# In[2]:


print(weather.describe())
print("-----------------------------Dataframe info----------------------------------")
print(weather.info())
print("--------------------------NULL sum-------------------------------------")
print(weather.isnull().sum())
weather.head(10)
# No nulls in weather dataframe
# Temperature have negative valuse as seen from min row
# date is an object


# In[3]:


print(fuel.describe())
print("-----------------------------Dataframe info----------------------------------")
print(fuel.info())
print("--------------------------NULL sum-------------------------------------")
print(fuel.isnull().sum())
fuel.head(10)
# No nulls in fuel dataframe
# no negative valuse in Fuel_Price


# In[4]:


sales[sales["Weekly_Sales"]<=0]
# there is 1358 rows with neagtive valuse and zeros in the dataframe which isn't a large number
# considring the dataframe size


# In[5]:


weather[weather["Temperature"]<=0]
# there is only 4 negative values in weatherdataframe


# In[6]:


# dropping the negative values and leaving the zeros
sales.drop(sales[sales['Weekly_Sales'] < 0].index, inplace = True)
weather.drop(weather[weather['Temperature'] < 0].index, inplace = True)


# In[7]:


merged =sales.merge(weather,on=['Store','Date'])
merged=merged.merge(fuel,on=['Store','Date'])
# turning the date from object to datetime and sorting the dataframe by it
merged['Date'] = pd.to_datetime(merged['Date'])
merged.sort_values(by=["Date"], inplace=True)
merged


# In[8]:


# 1
import seaborn as sns
plt.figure(figsize = (15,8))
sns.lineplot(x= merged['Date'],y=merged['Weekly_Sales'] )
plt.title('Weekly Sales over time', fontsize=16)
plt.show()


# In[9]:


# 2
# assuming that the brands are the stores
plt.figure(figsize = (15,8))
merged.groupby('Store').sum()['Weekly_Sales'].plot.bar()
plt.xticks(rotation=0)
plt.title('brands sells', fontsize=16)
plt.ylabel('Weekly_Sales', fontsize=16)
plt.xlabel('Store', fontsize=16)
# we see that store 20 and store 4 are the highest values


# In[10]:


#3 top ten selling stores
Top_stores=merged.groupby('Store').sum()['Weekly_Sales'].nlargest(n=10)
Top_stores


# In[11]:


# 4
Top_stores.plot.hist()
plt.xticks(rotation=0)
plt.title('Top 10 store sales', fontsize=16)
plt.xlabel('Weekly_Sales', fontsize=16)


# In[12]:


# 5
fig, ax = plt.subplots(figsize=(15,8))
subsetDataFrame = merged[merged['Store'].isin([1,2,4,6,10,13,14,20,27,39]) ]
ax = sns.barplot(x="Store", y="Weekly_Sales", hue="Holiday", data=subsetDataFrame,palette= "rocket_r")
ax.set_xlabel(xlabel = "Top 10 stores ", fontsize = 16)
ax.set_ylabel(ylabel = "Avg Weekly Sales", fontsize = 16)
plt.title('average weekly sales for the top ten selling stores during holidays and non-holidays', fontsize=16)
# جبت المحلات اللي مجموع المبيعات بتاعتها من التوب 10 وحطيتهم في داتا فريم وحسبتلهم المتوسط عشان اعرف اجيب قيمة المبيعات بتاعتهم وهى متقسمة
# على ايام العطلات 
# من الرسمة واضح ان المبيعات بتبقى اعلى في ايام العطلة 


# In[13]:


# 6
fig, ax = plt.subplots(figsize=(100,15))
ax = sns.barplot(x="Category", y="Weekly_Sales",hue="Store", data=subsetDataFrame)
plt.title('the average weekly sales for each brand department for the top 10 selling stores', fontsize=16)
ax.set_xlabel(xlabel = "Category", fontsize = 16)
ax.set_ylabel(ylabel = "Weekly_Sales", fontsize = 16)
# we aren't sure what is wanted here but we assumed that the brand department is the categories so we made this plot 
# that displays the weekly sales for each category in each store of the top ten stores
# بمعنى ان الرسمة ديه لو عاوز يعرف كل كاتيجوري بيساهم بقد ايه في المبيعات في كل ستور من ال10


# In[14]:


# 6
fig, ax = plt.subplots(figsize=(25,15))
ax = sns.barplot(x="Category", y="Weekly_Sales", data=subsetDataFrame)
plt.title('the average weekly sales for each brand department for the top 10 selling stores', fontsize=16)
ax.set_xlabel(xlabel = "Category", fontsize = 16)
ax.set_ylabel(ylabel = "Weekly_Sales", fontsize = 16)
# this is another solution for the previous question in case of it just wants to know the weekly sales of the categories 
# without including the stores in the plot
# لكن هنا الرسمة بتوضح مبيعات كل كاتيجوري في كل الستورز 


# In[15]:


# 7
fig, ax = plt.subplots(figsize=(12,6))
lineplot = sns.lineplot(x=merged['Date'], y=merged['Weekly_Sales'], data=merged, 
                        label = 'Weekly_Sales', legend=False)
sns.despine()
plt.ylabel('Weekly_Sales')
plt.title('the relationship between weekly sales and weather Temperature');

ax2 = ax.twinx()
lineplot2 = sns.lineplot(x=merged['Date'], y=merged['Temperature'], ax=ax2, color="r", 
                         label ='Temperature', legend=False) 
plt.ylabel('Temperature')
ax.figure.legend();


# In[16]:


# 8
fig, ax = plt.subplots(figsize=(12,6))
lineplot = sns.lineplot(x=merged['Date'], y=merged['Weekly_Sales'], data=merged, 
                        label = 'Weekly_Sales', legend=False)
sns.despine()
plt.ylabel('Weekly_Sales')
plt.title('the relationship between the cost of fuel and weekly sales');

ax2 = ax.twinx()
lineplot2 = sns.lineplot(x=merged['Date'], y=merged['Fuel_Price'], ax=ax2, color="r", 
                         label ='Fuel_Price', legend=False) 
plt.ylabel('Fuel_Price')
ax.figure.legend();


# In[17]:


# 9 only numeric comlumns
sns.pairplot(merged.drop(merged.columns[[4]], axis=1), diag_kind='kde', plot_kws={'alpha': 0.2}) 
# dropped the holiday columns becaues it was causing a problem because it's a boolean


# In[18]:


# 10
matrix = merged.corr().round(2)
plt.figure(figsize = (15,8))
mask = np.triu(np.ones_like(matrix, dtype=bool))
sns.heatmap(matrix, annot=True,vmax=1, vmin=-1, center=0, cmap='Purples', mask=mask)
plt.show()
plt.savefig('corelation matrix.png')
# from the matrix we can see that the weekly sales have a good relation with the category with 0.15 ,Holiday and temperature 
# have a strong negative relation with -0.16 but temperature have a good relation with fuel price with 0.14, all the remaining relations
# are weak,but all the the relations in general are weak as no value comes near 0.7


# In[51]:


from matplotlib import cm
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
X = merged[['Category','Holiday','Store']]  
y = merged['Weekly_Sales']
# I chose category and holiday because they have a good relation with weekly sales 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[63]:


knn_reg_model = KNeighborsRegressor(n_neighbors=10,algorithm='auto').fit(X_train,y_train)
# KNeighborsRegressor beacuse we are trying to forecast a continous variable


# In[64]:


knn_reg_model_prediction = knn_reg_model.predict([[1,True,4]])
knn_reg_model_prediction


# In[65]:


classification_score=knn_reg_model.score(X_test, y_test)*100
print("%",classification_score)


# In[66]:


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)


# In[67]:


LinearRegression_prediction = model.predict([[1,True,4]])
LinearRegression_prediction


# In[68]:


regression_score=model.score(X_test, y_test)*100
print("%",regression_score)


# In[69]:


new_df = merged[['Category','Weekly_Sales']]  
cols = new_df.columns
from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

new_df = ms.fit_transform(new_df)
new_df


# In[70]:


new_df = pd.DataFrame(new_df, columns=[cols])
new_df.head()


# In[71]:


from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(new_df)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.ylabel('CS')
plt.xlabel('Number of clusters')
plt.show()
# from the elbow plot we can see that the optimal number of clusters is 3


# In[72]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0) 

kmeans.fit(new_df)


# In[73]:


kmeans = KMeans(n_clusters=3).fit(new_df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(new_df['Category'], new_df['Weekly_Sales'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()


# In[80]:




kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(new_df)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[79]:




kmeans = KMeans(n_clusters=5, random_state=0)

kmeans.fit(new_df)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

# 5 cluster is a better number although the accuracy still 0


# In[ ]:




