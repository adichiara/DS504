# Databricks notebook source
import logging 
logging.getLogger("py4j").setLevel(logging.ERROR)



# COMMAND ----------

import numpy as np
import pandas as pd

!pip install requests
!pip install urllib

import requests
import urllib
from urllib.parse import urlparse

import datetime
from datetime import datetime

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

spark

# COMMAND ----------

# urllib.request.urlretrieve('https://raw.githubusercontent.com/adichiara/DS504/main/source_data.csv', '/dbfs/saved_df/source_data.csv')

# COMMAND ----------

# MAGIC %sh -e rm -r DS504  

# COMMAND ----------

# MAGIC %sh git clone https://github.com/adichiara/DS504

# COMMAND ----------

# MAGIC %sh -e ls DS504/ -al 

# COMMAND ----------

# MAGIC %fs mv 'file:/databricks/driver/DS504/source_data.csv' '/saved_df/source_data.csv'

# COMMAND ----------

source_df = pd.read_csv("/dbfs/saved_df/source_data.csv", header=0)
display(source_df)

# COMMAND ----------

source_id_list = list(source_df['source_id'])


# COMMAND ----------

source_id_list = ['abc-news ',
                  'associated-press ',
                  'axios ',
                  'bloomberg ',
                  'breitbart-news ',
                  'buzzfeed ',
                  'cbc-news ',
                  'cnn ',
                  'fox-news',
                  'msnbc ',
                  'national-review ',
                  'nbc-news ',
                  'politico ',
                  'reuters ',
                  'the-hill ',
                  'the-huffington-post ',
                  'the-wall-street-journal ',
                  'the-washington-post ',
                  'the-washington-times ',
                  'usa-today ']

source_id_str = str(source_id_list)[1:len(str(source_id_list))-1].replace("'","")
print(source_id_str)


# COMMAND ----------


url = 'http://newsapi.org/v2/everything?'

def getArticleData(q, source_id, from_date, to_date, limit):
  
  search_params = {
    'language':'en',
    'sortBy':'popularity',
    'apiKey':newsAPI_key
  }       

  if (q):
    search_params['q'] = q
  if (source_id):
    search_params['sources'] = source_id
  if (from_date):
    search_params['from'] = from_date
  if (to_date):
    search_params['to'] = to_date
  if (limit):
    search_params['pageSize'] = limit
    
  response = requests.get(url, search_params)
  df = pd.DataFrame()
  
  if (response.json()['status']=='error'):
    print(response.json()['code'],response.json()['message'])
    return(df)
  else:    
    articles_json = response.json()['articles']
    if (bool(response.json()['articles'])):
      for x in response.json()['articles']:
        article_dict = {}
        try:
          article_dict['source_name'] = x['source']['name']
        except:
          pass
        try:
          article_dict['source_id'] = x['source']['id']
        except:
          pass
        try:
          date = datetime.strptime(x['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
          article_dict['date'] = str(date.year) + '-' + str(date.month) + '-' + str(date.day)
          article_dict['year'] = date.year
          article_dict['month'] = date.month
          article_dict['day'] = date.day
        except:
          pass
        try:
          article_dict['title'] = x['title']
        except:
          pass
        try:
          article_dict['domain'] = urlparse(x['url']).netloc
        except:
          pass
        try:
          article_dict['urlToImage'] = x['urlToImage']
        except:
          continue
          
        df = df.append(article_dict, ignore_index=True)

  return(df)


# COMMAND ----------

temp_dict = {"olddate":"2020-10-21T06:12:16Z"}


date = datetime.strptime(temp_dict['olddate'], '%Y-%m-%dT%H:%M:%SZ')

temp_dict['date'] = str(date.year) + '-' + str(date.month) + '-' + str(date.day)
temp_dict['year'] = date.year
temp_dict['month'] = date.month
temp_dict['day'] = date.day
temp_dict['hour'] = date.hour
temp_dict['minute'] = date.minute

temp_dict


# COMMAND ----------


newsAPI_key = '802c3d598bfe418f9165b8a1d655479e'

oldest = str(datetime.today().year) + '-' + str(datetime.today().month) + '-' + str(datetime.today().day-7)
newest = str(datetime.today().year) + '-' + str(datetime.today().month) + '-' + str(datetime.today().day)

temp_df = getArticleData(source_id='cnn', q='biden', from_date=oldest, to_date=newest, limit=1)


# COMMAND ----------


# newsAPI_key = '467ee87cae8346669f5cf44c9e9a5c57'
# newsAPI_key = 'a28ebf83077c40158d5b9d19cbaa97f3'
newsAPI_key = '05bbf397f9b04b8185dcb00b77f1c429'
# newsAPI_key = '802c3d598bfe418f9165b8a1d655479e'
# newsAPI_key = 'fbd13bdbb4d74fddbd53365207fdadad'
# newsAPI_key = 'ab5008e3358b4e188019c24a468df86a'
# newsAPI_key = '294e9cf5c8ba4f328c9dc35b75c4383c'


q_list = ['trump','biden']

date_range = 7   # how many recent days to include in search results

oldest = str(datetime.today().year) + '-' + str(datetime.today().month) + '-' + str(datetime.today().day - date_range)
newest = str(datetime.today().year) + '-' + str(datetime.today().month) + '-' + str(datetime.today().day)

article_df = pd.DataFrame()

for query in q_list:
  for s in source_id_list:
    article_df = article_df.append(getArticleData(source_id=s, 
                                                  q=query, 
                                                  from_date=oldest, 
                                                  to_date=newest, 
                                                  limit=50))

display(article_df)


# COMMAND ----------

article_df = article_df.drop_duplicates('title') # drop duplicate articles
article_df = article_df.loc[article_df.urlToImage != 'null',:] # drop null URL strings
article_df = article_df.reset_index(drop=True)


# COMMAND ----------

article_df.to_csv('/dbfs/saved_df/article_df.csv')

# COMMAND ----------

test_df = pd.read_csv("/dbfs/saved_df/article_df.csv", header=0, index_col=0)
display(test_df)

# COMMAND ----------

# Upload datafile to github repo

!pip install PyGithub
from github import Github

# ---------

git_file = 'article_df.csv'
dbfs_file = '/dbfs/saved_df/article_df.csv'

# ---------

f = open("github_token.txt", "r")
github_token = f.read()
f.close()

g = Github(github_token)
repo = g.get_repo("adichiara/DS504")
contents = repo.get_contents("")
all_files = []

while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    else:
        file = file_content
        all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))


with open(dbfs_file, 'r') as file:
    content = file.read()

# ---------
    
commit_txt = "uploaded from Databricks."

if git_file in all_files:
    contents = repo.get_contents(git_file)
    repo.update_file(contents.path, commit_txt, content, contents.sha, branch="main")
    print(git_file + ' UPDATED')
else:
    repo.create_file(git_file, commit_txt, content, branch="main")
    print(git_file + ' CREATED')
    
    