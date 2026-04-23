#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
print(sklearn.__version__)


# In[2]:


import pandas as pd
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
print(df.shape)
print(df.head())


# In[3]:


print(df['label'].value_counts())


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay


# In[5]:


df['label_num'] = df['label'].map({'ham':0,'spam':1})


# In[6]:


X = df['message']
y = df['label_num']


# In[7]:


X_train , X_test , y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


# In[8]:


vectorizer = TfidfVectorizer(stop_words = 'english',max_features = 5000)
X_Train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[9]:


model = MultinomialNB()
model.fit(X_Train_tfidf,y_train)
y_pred = model.predict(X_test_tfidf)


# In[10]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[11]:


def predict_message(message):
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    return "SPAM" if prediction == 1 else "HAM "

# Test it!
msg = input("Enter the msg to predict:");
predict_message(msg)


# In[12]:


import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])

disp.plot(cmap=plt.cm.Reds)
plt.show()


# In[ ]:




