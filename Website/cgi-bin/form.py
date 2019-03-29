#!/usr/bin/env python
# coding: utf-8

# In[5]:


# import sys
# sys.path.append("D:\python\lib\site-packages")

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import pandas as pd

from sklearn import preprocessing, svm, utils
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier


# In[6]:


data_app = pd.read_csv('./google-play-store-apps/googleplaystore.csv', sep=',')


# In[7]:


pd.read_csv('./google-play-store-apps/googleplaystore.csv', sep=',',encoding="ISO-8859-1")


# In[8]:


data_app = pd.read_csv('./google-play-store-apps/googleplaystore.csv', sep=',')
print(data_app.shape)

head = data_app.columns
print(head)


# In[ ]:





# In[9]:


print(data_app.shape)

head = data_app.columns
print("----")
for i in range(len(head)):
    print(i, head[i])
print("----")

data_app_c = np.array(data_app)

i=0
for i in range(len(data_app_c[:])):
    if data_app_c[i][5] == "Free":
        print("THIS: ", data_app_c[i])
        data_app_c1 = np.delete(data_app_c, i, 0)

data_app_c = data_app_c1


i=0
for i in range(len(data_app_c[:])):
    for char in data_app_c[i][5]:
        if char in " ?.!/;:+,":
            data_app_c[i][5] = data_app_c[i][5].replace(char,'')
    if float(data_app_c[i][5]) <= 100:
        data_app_c[i][5] = 100
    if float(data_app_c[i][5]) > 100 and int(data_app_c[i][5])<= 5000:
        data_app_c[i][5] = 5000
    if float(data_app_c[i][5]) > 5000 and int(data_app_c[i][5])<= 50000:
        data_app_c[i][5] = 50000
    if float(data_app_c[i][5]) > 50000 and int(data_app_c[i][5])<= 500000:
        data_app_c[i][5] = 500000
    if float(data_app_c[i][5]) > 500000 and int(data_app_c[i][5])<= 5000000:
        data_app_c[i][5] = 5000000
    if float(data_app_c[i][5]) > 5000000:
        data_app_c[i][5] = 10000000
        
for i in range(len(data_app_c[:])):
    if data_app_c[i][4].endswith("M"):
        data_app_c[i][4] = data_app_c[i][4][:-1]
    if data_app_c[i][4] == "Varies with device":
        data_app_c[i][4] = float("NaN")
    elif data_app_c[i][4].endswith("k"):
        data_app_c[i][4] = float(data_app_c[i][4][:-1])/1000
    if data_app_c[i][2] == "#¡NUM!":
        data_app_c[i][2] = float("NaN")
    if data_app_c[i][6] == 'Free':     #free is encoded as 0
        data_app_c[i][6] = 0
    if data_app_c[i][6] == 'Paid':     #paid is encoded as 1
        data_app_c[i][6] = 1 
 
print(data_app_c[2])


# In[10]:


data_app_cN = pd.DataFrame(data_app_c)
data_app_cN = data_app_cN .dropna()

data_app_cN = data_app_cN.values
print(data_app_c.shape)
print(data_app_cN.shape)

print(data_app_c[2])
print(data_app_cN[2])


# In[11]:


#transforming numerical values to float 
for i in range(len(data_app_c[:])):
    data_app_c[i][2] = float(data_app_c[i][2]) #Rating
    data_app_c[i][3] = float(data_app_c[i][3]) #Reviews
    data_app_c[i][4] = float(data_app_c[i][4]) #Size
    data_app_c[i][5] = int(data_app_c[i][5]) #Installs

        
#delete the name category for everything
del_categories = [0, 7, 9, 10, 11, 12]
data_app_c = np.delete(data_app_c, del_categories, 1)
data_app_cN = np.delete(data_app_cN, del_categories, 1)
head = np.delete(head, del_categories)

print(data_app_c.shape)
print(data_app_cN.shape)
print("----")


print("----")
for i in range(len(head)):
    print(i, head[i])
print("----")

# columns = [1'Category', 2'Rating', 3'Reviews', 4'Size', 5 Installs', 6'Type', 7'Content Rating'])


# In[12]:


# # print(data_app_c.isna().sum())

# # print(data_app_c.isna().sum())
# data_app_c.isnull().sum().sum()
# data_app_c = data_app_c.dropna()
# print(data_app_c.isnull().sum().sum())
# data_app_c = data_app_c.reset_index()
# data_app_c = data_app_c.values


# In[13]:


le = preprocessing.LabelEncoder()

a = le.fit_transform(data_app_cN.T[0].astype(str)) #Category
data_app_cN.T[0] = a
a = le.fit_transform(data_app_cN.T[5].astype(str)) #Type
data_app_cN.T[5] = a
a = le.fit_transform(data_app_cN.T[6].astype(str)) #Content Rating
data_app_cN.T[6] = a


#transforming numerical values to float 
for i in range(len(data_app_cN[:])):
    for h in range(len(head)):
        data_app_cN[i][h] = float(data_app_cN[i][h])

#transforming numerical values to float 
for i in range(len(data_app_cN[:])):
    data_app_cN[i][4] = int(data_app_cN[i][4])

# a = le.fit_transform(data_app_cN.T[0].astype(str)) #Category
# data_app_cN.T[0] = a
# a = le.fit_transform(data_app_cN.T[5].astype(str)) #Type
# data_app_cN.T[5] = a
# a = le.fit_transform(data_app_cN.T[6].astype(str)) #Content Rating
# data_app_cN.T[6] = a


# enc = preprocessing.OneHotEncoder()
# enc.fit(data_app_cN)
# onehotlabels = enc.transform(data_app_cN).toarray()
# onehotlabels.shape

# print(data_app_cN.shape)
# print(data_app_cN)
# print(onehotlabels.shape)
# print(onehotlabels)


print(data_app_cN)
print(data_app_cN.shape)


# In[ ]:





# In[ ]:





# In[14]:


# split into input (X) and output (Y) variables
Xa = data_app_cN[:, [1, 2]]
# Xa = data_app_cN[:, [6]]
Ya = data_app_cN[:,4]

X_train, X_test, y_train, y_test = train_test_split(Xa, Ya, test_size=0.33, random_state=42)

print(X_train, X_test, y_train, y_test)

#print(utils.multiclass.type_of_target(y_train.astype('int')))

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train.astype('float'), y_train.astype('int'))
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


# In[15]:


pred = clf.predict(X_test.astype('float'))
print(X_test.shape)

print("Accuracy score: ", accuracy_score(y_test.astype('int'), pred.astype("int")))

install_win = np.unique(y_train.astype('int')).astype("str")
print(install_win)

print(classification_report(y_test.astype('int'), pred.astype("int"), target_names=install_win, sample_weight=None, digits=3))


# In[16]:


gnb = GaussianNB()
KNN = KNeighborsClassifier(n_neighbors=1)
BNB = BernoulliNB()
LR = LogisticRegression()
SVC = SVC()
SDG = SGDClassifier()
LSVC = LinearSVC()

x1, x2, y1, y2 = X_train, X_test, y_train, y_test

# Train our classifier and test predict
gnb.fit(x1.astype("float"), y1.astype("int"))
y2_GNB_model = gnb.predict(x2.astype("float"))
print("GaussianNB Accuracy :", accuracy_score(y2.astype("int"), y2_GNB_model.astype("int")))

KNN.fit(x1.astype("float"),y1.astype("int"))
y2_KNN_model = KNN.predict(x2.astype("float"))
print("KNN Accuracy :", accuracy_score(y2.astype("int"), y2_KNN_model.astype("int")))

BNB.fit(x1.astype("float"),y1.astype("int"))
y2_BNB_model = BNB.predict(x2.astype("float"))
print("BNB Accuracy :", accuracy_score(y2.astype("int"), y2_BNB_model.astype("int")))

LR.fit(x1.astype("float"),y1.astype("int"))
y2_LR_model = LR.predict(x2.astype("float"))
print("LR Accuracy :", accuracy_score(y2.astype("int"), y2_LR_model.astype("int")))

SDG.fit(x1.astype("float"),y1.astype("int"))
y2_SDG_model = SDG.predict(x2.astype("float"))
print("SDG Accuracy :", accuracy_score(y2.astype("int"), y2_SDG_model.astype("int")))

SVC.fit(x1.astype("float"),y1.astype("int"))
y2_SVC_model = SVC.predict(x2.astype("float"))
print("SVC Accuracy :", accuracy_score(y2.astype("int"), y2_SVC_model.astype("int")))

LSVC.fit(x1.astype("float"),y1.astype("int"))
y2_LSVC_model = LSVC.predict(x2.astype("float"))
print("LSVC Accuracy :", accuracy_score(y2.astype("int"), y2_LSVC_model.astype("int")))


# In[24]:


#one line prediction


#!/usr/bin/env python3
import cgi


form = cgi.FieldStorage()
text1 = form["TEXT_1"].value
text2 = form["TEXT_2"].value

rev = float(text1)
rat = float(text2)
g=[rev,rat]
f = np.array(g)
magic = f.reshape(1,-1)
KNN.fit(x1.astype("float"),y1.astype("int"))
y2_KNN_model = KNN.predict(magic.astype("float"))

In[11]:
print("Content-type: text/html\n")
print("""<!DOCTYPE HTML>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Installs prediction</title>
        </head>
        <body>""")

print("<h1>Installs prediction:</h1>")
print("<p>Installs: {}</p>".format(y2_KNN_model1))

print("""</body>
        </html>""")


# In[ ]:


# # fix random seed for reproducibility
# numpy.random.seed(7)


# In[ ]:


# # split into input (X) and output (Y) variables
# X = data_app_cN[:, [0, 1, 2, 3, 5, 6]]
# Y = data_app_cN[:,4]


# In[ ]:


# # create model
# model = Sequential()
# model.add(Dense(12, input_dim=7, init='uniform', activation='relu'))
# model.add(Dense(7, init='uniform', activation='relu'))
# model.add(Dense(1, init='uniform', activation='sigmoid'))
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
# model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# # calculate predictions
# predictions = model.predict(X)
# # round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)


# In[ ]:





# In[ ]:





# In[ ]:


# pd.DataFrame(onehotlabels)


# In[ ]:


# #print(pd.DataFrame.describe(pd.DataFrame(data_app_c[:, 4])))
# #print("--------")
# n_samp = 7000

# data_app_c = onehotlabels

# random_idx = np.random.randint(0, len(data_app_c[:, 1]), n_samp)
# rating_t = data_app_c[:, 1][random_idx].astype(float)

# random_idx = np.random.randint(0, len(data_app_c[:, 2]), n_samp)
# review_t = data_app_c[:, 2][random_idx].astype(float)

# random_idx = np.random.randint(0, len(data_app_c[:, 4]), n_samp)
# installs_t = data_app_c[:, 4][random_idx].astype(int)





# plt.plot(review_t, alpha = 0.5, label = head[2])
# plt.title(head[2])

# plt.plot(installs_t, alpha = 0.5, label = head[4])
# plt.title(head[4])

# plt.plot(rating_t, alpha = 0.5, label = head[1])
# plt.title(head[4])

# plt.legend()


# #not sure what this should plot but it gives an error
# plt.figure(figsize=(10,10))
# 
# i=0
# for i in enumerate(head):        
#     plt.subplot(5,5, i+1)
#     plt.hist(data_app_c[:,i])
#     plt.title(head[i])
#     plt.legend()
#     plt.tight_layout()

# In[ ]:


# dataframe_app_c = pd.DataFrame(onehotlabels)
# dataframe_app_c.shape


# In[ ]:


# K-Means Clustering

# X = dataframe_app_c


# #REMOVED AOVE: App  object

# #Category           object
# #Rating            float64
# #Reviews            object
# #Size               object
# #Installs           object
# #Type               object
# #Price              object
# #Content Rating     object
# #Genres             object
# #Last Updated       object
# #Current Ver        object
# #Android Ver        object

# X = pd.DataFrame(X)
# X = X.convert_objects(convert_numeric=True)
# #X.columns = onehotlabels.T[:] #['Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type','Content Rating']

# # Eliminating null values
# #for i in X.columns:
#     #X[i] = X[i].fillna(int(X[i].mean()))
# #for i in X.columns:
#     #print(X[i].isnull().sum())
    

# # Using the elbow method to find  the optimal number of clusters
# wcss = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1,11),wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# # Applying k-means to the googleplaydata dataset
# kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0) 
# y_kmeans = kmeans.fit_predict(X)

# X = X.as_matrix(columns=None)

# # Visualising the clusters
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],s=100,c='red',label='C1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],s=100,c='blue',label='C2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1],s=100,c='green',label='C3')
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
# plt.title('Clusters of apps')
# plt.legend()
# plt.show()


# In[ ]:


# from __future__ import absolute_import, division, print_function

# # TensorFlow and tf.keras
# import tensorflow as tf
# from tensorflow import keras

# # Helper libraries
# import numpy as np
# import matplotlib.pyplot as plt

# print(tf.__version__)


# In[ ]:




