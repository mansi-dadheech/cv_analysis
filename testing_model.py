#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import os
import sys
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher


# In[ ]:
file_arg = sys.argv[1]
# In[2]:


#parsing pdf to text
from tika import parser
raw = parser.from_file(file_arg)
text = raw['content']


# In[3]:


text


# In[4]:


#function that does phrase matching and builds a candidate profile
def create_profile(file):
  #below is the csv where we have all the keywords, you can customize your own
        keyword_dict = pd.read_csv('..\skills_classify.csv',encoding='latin1')
        language = [nlp(text) for text in keyword_dict['Languages'].dropna(axis = 0)]
        big_data = [nlp(text) for text in keyword_dict['BigDataAnalysis'].dropna(axis = 0)]
        coding = [nlp(text) for text in keyword_dict['CodingAndProgramming'].dropna(axis = 0)]
        data_science = [nlp(text) for text in keyword_dict['DataScience'].dropna(axis = 0)]
        devops = [nlp(text) for text in keyword_dict['DevOps'].dropna(axis = 0)]
        cloud = [nlp(text) for text in keyword_dict['CloudComputing'].dropna(axis = 0)]
        ops = [nlp(text) for text in keyword_dict['OperatingSystem'].dropna(axis = 0)]
        web = [nlp(text) for text in keyword_dict['WebDevelopement'].dropna(axis = 0)]
        dbms = [nlp(text) for text in keyword_dict['DBMS'].dropna(axis = 0)]
        app = [nlp(text) for text in keyword_dict['AppDevelopment'].dropna(axis = 0)]
        security = [nlp(text) for text in keyword_dict['Security'].dropna(axis = 0)]
        other = [nlp(text) for text in keyword_dict['Others'].dropna(axis = 0)]

        matcher = PhraseMatcher(nlp.vocab)
        matcher.add('Languages', None, *language)
        matcher.add('BigDataAnalysis', None, *big_data)
        matcher.add('CodingAndProgramming', None, *coding)
        matcher.add('DataScience', None, *data_science)
        matcher.add('DevOps', None, *devops)
        matcher.add('CloudComputing', None, *cloud)
        matcher.add('OperatingSystem', None, *ops)
        matcher.add('WebDevelopement', None, *web)
        matcher.add('DBMS', None, *dbms)
        matcher.add('AppDevelopment', None, *app)
        matcher.add('Security', None, *security)
        matcher.add('Others', None, *other)
        doc = nlp(text)
        #print(doc)
        d = []  
        matches = matcher(doc)
        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
            span = doc[start : end]  # get the matched slice of the doc
            d.append((rule_id, span.text))      
        keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())


      ## convertimg string of keywords to dataframe

        df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
        df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
        df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
        df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
        df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
        #print(df1['Subject'])


        base = os.path.basename(file)
        filename = os.path.splitext(base)[0]

        name = filename.split('_')
        name2 = name[0]
        name2 = name2.lower()
      ## converting str to dataframe
        name3 = pd.read_csv(StringIO(name2),names = ['CandidateName'])

        dataf = pd.concat([name3['CandidateName'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
        dataf['CandidateName'].fillna(dataf['CandidateName'].iloc[0], inplace = True)

        return(dataf)

    #function ends


# In[5]:


final_database=pd.DataFrame()
dat = create_profile(file_arg)
final_database = final_database.append(dat)


# In[6]:


final_database
subcol = final_database["Keyword"].unique()

# In[7]:


keyword_dict = pd.read_csv('..\skills_classify.csv',encoding='latin1')


# In[8]:


final_database2 = final_database['Keyword'].groupby([final_database['CandidateName'], final_database['Subject']]).count().unstack()
final_database2.reset_index(inplace = True)
final_database2.fillna(0,inplace=True)
if(len(final_database2.iloc[0]) < len(keyword_dict.iloc[0])):
    i=0
    for i in keyword_dict:
        if i not in final_database2:
            final_database2[i] = 0

new_data = final_database2.iloc[:,1:]
new_data.index = final_database2['CandidateName']
#execute the below line if you want to see the candidate profile in a csv format
sample2=new_data.to_csv('..\\testingData.csv')


# In[9]:


df = pd.read_csv('..\\testingData.csv')


# In[10]:


df


# # Extracting CGPA

# In[11]:


from tika import parser
raw = parser.from_file(file_arg)
text = raw['content']


# In[12]:


text


# In[13]:


text = text.lower()


# In[14]:


import re
if (text.find('cgpa') != -1):
    index = text.index('cgpa')
    text = text.strip(' ') 
    #print(text)
    #num = re.findall(r'\d+.\d+', text[index-10:len(text)])
    num = re.findall(r'\d+(?:\.\d*)', text[index-10:len(text)])
else:
    num = [0]
    


# In[15]:


num[0]


# In[16]:


df['CGPA'] = num[0]


# In[17]:


df



# In[18]:

name = df['CandidateName'][0]
X_test =df.drop(['CandidateName'],axis=1)


# In[19]:


import pickle


# In[20]:


loaded_model = pickle.load(open("..\cv_analysis_model.sav", 'rb'))
result = loaded_model.predict(X_test)
if result == 0:
    df['Placement_status'] = result
    print("Sorry " + str(name) +  "!  Work on your skills")
else:
    df['Placement_status'] = result
    print("Great Work "  + str(name) +  "!  You have Amazing SkillSet")


# In[21]:

from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, y_pred)*100)

df


# In[22]:


df1 = pd.read_csv("..\\finalDatasets.csv")


# In[23]:


df1.columns


# In[24]:


df1 = df1.append(df)


# In[25]:


df1


# In[26]:


from sklearn.impute import SimpleImputer
def imputation(df1,variable,median):
    df1[variable] = df1[variable].fillna(median)


# In[27]:


median = df1.CGPA.median()


# In[28]:


imputation(df1,'CGPA',median)


# In[29]:


df1.to_csv("..\\finalDatasets.csv",index= False)
df1 = pd.read_csv("..\\finalDatasets.csv")

# In[30]:


from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
object_cols=['Placement_status']

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    df1[col] = label_encoder.fit_transform(df1[col])
df1


# In[31]:


X=df1.drop(['Placement_status','CandidateName'],axis=1)
y=df1.Placement_status


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=1)


# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[34]:
#print("After retraining...")

from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, y_pred)*100)


# In[ ]:


filename = "..\cv_analysis_model.sav"
pickle.dump(model, open(filename, 'wb'))

df = df.drop(['CodingAndProgramming','DataScience','DevOps','Languages','OperatingSystem', 'WebDevelopement','AppDevelopment','BigDataAnalysis', 'CloudComputing', 'DBMS', 'Others', 'Security'],axis=1)
subcol = subcol.tolist()
subcoldf = pd.DataFrame({"SkillSet" : [subcol]})
df["SkillSet"] = subcoldf["SkillSet"]
df.to_csv('cdata.csv', mode='a' , header=False, index = False)

os.system("python ..\\featureplots.py")