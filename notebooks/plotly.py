# Databricks notebook source
import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

!pip install chart_studio
!pip install plotly --upgrade

import chart_studio
import chart_studio.plotly as py
from chart_studio.plotly import plot, iplot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

!pip install scikit-image
import skimage
from skimage import io

from datetime import datetime

chart_studio.tools.set_credentials_file(username='adamjd', 
                                        api_key='1piYaG7ohT90yKxLEjL7')




# COMMAND ----------

# make backups of existing data 

dbutils.fs.cp('/saved_df/fer_df.csv', '/archive/fer_df.csv')
dbutils.fs.cp('/saved_df/article_totals.csv', '/article_totals.csv')


# COMMAND ----------

face_df = pd.read_csv("/dbfs/saved_df/face_df.csv", header=0, index_col=0)
source_df = pd.read_csv("/dbfs/saved_df/source_data.csv", header=0, index_col=0)
article_df = pd.read_csv("/dbfs/saved_df/article_df.csv", header=0, index_col=0)

# COMMAND ----------

face_np = np.load('/dbfs/faces/face_np.npy', allow_pickle=True)

# COMMAND ----------

fer_df = pd.merge(face_df, source_df, left_on=['source'], right_on=['source_name'])
fer_df = fer_df.loc[fer_df.emotions,:]


# COMMAND ----------

article_totals = article_df.groupby('source_name', as_index=False).count()
article_totals = article_totals.filter(['source_name', 'urlToImage'], axis=1)
article_totals = article_totals.rename(columns={'urlToImage':'num_articles'})
fer_df = pd.merge(fer_df, article_totals, left_on=['source'], right_on=['source_name'])

# COMMAND ----------

fer_df['top_emotion'] = face_df[['happy','sad','angry','surprise','fear','disgust','neutral']].idxmax(axis=1)
fer_df['top_emotion_prob'] = fer_df[['happy','sad','angry','surprise','fear','disgust','neutral']].max(axis=1)

fer_df.loc[fer_df['top_emotion']=='happy', 'top_happy'] = 1
fer_df.loc[fer_df['top_emotion']=='sad', 'top_sad'] = 1
fer_df.loc[fer_df['top_emotion']=='angry', 'top_angry'] = 1
fer_df.loc[fer_df['top_emotion']=='surprise', 'top_surprise'] = 1
fer_df.loc[fer_df['top_emotion']=='fear', 'top_fear'] = 1
fer_df.loc[fer_df['top_emotion']=='disgust', 'top_disgust'] = 1
fer_df.loc[fer_df['top_emotion']=='neutral', 'top_neutral'] = 1


# COMMAND ----------

article_totals.to_csv('/dbfs/saved_df/article_totals.csv')
fer_df.to_csv('/dbfs/saved_df/fer_df.csv')


# COMMAND ----------

timestamp = datetime.today()
timestamp_str = "Updated at: " + str(timestamp.month) + "/" + str(timestamp.day) + "/" + str(timestamp.year) + " " + str(timestamp.hour) + ":" + str(timestamp.minute) + " UTC"


# COMMAND ----------


df1 = article_totals

fig = px.bar(df1, 
             y='source_name',
             x='num_articles',             
             title='How many total news articles were collected from each news source?<br>' + timestamp_str,
             labels=dict(source_name="News Source", num_articles="Total Articles Collected in Latest Query"),     
             height=600, width=600)

fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()

py.iplot(fig, filename='total-articles-by-source')

# COMMAND ----------


df1 = fer_df
df1.sort_values(['name'], ascending=False, inplace=True)

num_articles_order = list(article_totals.sort_values('num_articles', ascending=False)['source_name'])


fig = px.bar(df1, 
             y='source',
             color='name',
             title='How many faces were detected in the images collected from each news source<br>'  + timestamp_str,
             labels=dict(name="", source="News Source", count="Number of Faces Detected in Collected Articles"),     
             category_orders={'source':num_articles_order},
             height=600, width=700)

# fig.update_layout(yaxis={'categoryorder':[num_articles_order]})
fig.show()
py.iplot(fig, filename='total-faces-detected-by-source')

# COMMAND ----------

date_list = np.sort(fer_df['date'].unique())

fig = px.bar(fer_df, 
             x="date", 
             color='name', 
             facet_col='source_bias',
             height=600, width=1400,
             labels=dict(name="", value="%"),
#              template="simple_white",
             title='Number of news articles in search query containing images of Donald Trump and Joe Biden (from past week).<br>' + timestamp_str,
             category_orders={"source_bias":['Left',
                                             'Lean Left',
                                             'Center',
                                             'Lean Right', 
                                             'Right'],
                             "date":date_list}
            )

 

fig.update_xaxes(tickformat="%m-%d", 
                 tickangle=-90,
                 title_text='',
                 type='category')

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.show()
py.iplot(fig, filename='articles-containing-trump-biden-photos-past-week')


# COMMAND ----------



df1 = pd.melt(fer_df, 
              id_vars=['name'],
               value_vars=['happy', 
                   'sad', 
                   'angry', 
                   'fear', 
                   'surprise', 
                   'disgust', 
                   'neutral'])


df1.loc[df1.variable=='happy', 'variable'] = 'Happy'
df1.loc[df1.variable=='sad', 'variable'] = 'Sad'
df1.loc[df1.variable=='angry', 'variable'] = 'Angry'
df1.loc[df1.variable=='fear', 'variable'] = 'Fear'
df1.loc[df1.variable=='surprise', 'variable'] = 'Surprise'
df1.loc[df1.variable=='disgust', 'variable'] = 'Disgust'
df1.loc[df1.variable=='neutral', 'variable'] = 'Neutral'


fig = px.box(df1, 
             y='value',
             x='variable',
             color='variable',
             height=600, width=700,
             title='What were the distributions of probabilities returned by the facial recognition<br>model for each type of facial expression?   ' + timestamp_str,
             labels=dict(name="", variable="", value="Predicted Probability of Emotion"),
             category_orders={"variable":['Happy','Sad','Angry','Fear','Surprise','Disgust','Neutral']})
                     
             
fig.show()
py.iplot(fig, filename='probability-boxplots')


# COMMAND ----------

df1 = fer_df

df1.loc[df1.top_emotion=='happy', 'top_emotion'] = 'Happy'
df1.loc[df1.top_emotion=='sad', 'top_emotion'] = 'Sad'
df1.loc[df1.top_emotion=='angry', 'top_emotion'] = 'Angry'
df1.loc[df1.top_emotion=='fear', 'top_emotion'] = 'Fear'
df1.loc[df1.top_emotion=='surprise', 'top_emotion'] = 'Surprise'
df1.loc[df1.top_emotion=='disgust', 'top_emotion'] = 'Disgust'
df1.loc[df1.top_emotion=='neutral', 'top_emotion'] = 'Neutral'


fig = px.box(df1, 
             y='top_emotion_prob',
             x='top_emotion',
#              color='top_emotion',
             height=600, width=700,
             title='What were the distributions for the highest predicted probability expression<br>for each face image (i.e., the "top emotion")?  ' + timestamp_str,
             labels=dict(top_emotion_prob="Probabilities of Top Expression for Each Face", top_emotion="Top Emotion"),
             category_orders={"top_emotion":['Happy','Sad','Angry','Fear','Surprise','Disgust','Neutral']})


fig.update_yaxes(range=[0,1])

py.iplot(fig, filename='top-emotion-probability-boxplots')
fig.show()


# COMMAND ----------



df1 = fer_df.groupby(['name','top_emotion'], as_index=False).sum()
df1.sort_values(['name'], ascending=False, inplace=True)

df1['top_emotion_sum'] = df1[['top_happy','top_sad','top_angry','top_surprise','top_fear','top_disgust','top_neutral']].max(axis=1)

df1.loc[df1.top_emotion=='happy', 'top_emotion'] = 'Happy'
df1.loc[df1.top_emotion=='sad', 'top_emotion'] = 'Sad'
df1.loc[df1.top_emotion=='angry', 'top_emotion'] = 'Angry'
df1.loc[df1.top_emotion=='fear', 'top_emotion'] = 'Fear'
df1.loc[df1.top_emotion=='surprise', 'top_emotion'] = 'Surprise'
df1.loc[df1.top_emotion=='disgust', 'top_emotion'] = 'Disgust'
df1.loc[df1.top_emotion=='neutral', 'top_emotion'] = 'Neutral'


fig = px.bar(df1, 
             x="name", 
             y="top_emotion_sum",
             color='top_emotion',
             text='top_emotion_sum',
             facet_col='top_emotion',    
             height=400, width=1200,
             title="How many face images returned by the search query were classified for each facial expression?<br>" + timestamp_str,
             labels=dict(name="", top_emotion_sum="Number of face images classified", top_emotion="Predicted Facial Expression"),     
#              template="simple_white",
             category_orders={"top_emotion":['Happy','Sad','Angry','Fear','Surprise','Disgust','Neutral']}
            )

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.show()
py.iplot(fig, filename='facial-expression-totals-barchart')



# COMMAND ----------



df1 = fer_df.groupby(['source_bias','name'], as_index=False).sum()
df1['happy_pct'] = df1.top_happy/df1.emotions
df1['sad_pct'] = df1.top_sad/df1.emotions
df1['angry_pct'] = df1.top_angry/df1.emotions
df1['fear_pct'] = df1.top_fear/df1.emotions
df1['surprise_pct'] = df1.top_surprise/df1.emotions
df1['disgust_pct'] = df1.top_disgust/df1.emotions
df1['neutral_pct'] = df1.neutral/df1.emotions


df1 = pd.melt(df1, 
              id_vars=['source_bias','name'],
              value_vars=['happy_pct', 
                   'sad_pct', 
                   'angry_pct', 
                   'fear_pct', 
                   'surprise_pct', 
                   'disgust_pct', 
                   'neutral_pct'])

df1 = df1.groupby(['source_bias','name','variable'], as_index=False).mean()

df1.sort_values(['name'], ascending=False, inplace=True)

df1.loc[df1.variable=='happy_pct', 'variable'] = 'Happy'
df1.loc[df1.variable=='sad_pct', 'variable'] = 'Sad'
df1.loc[df1.variable=='angry_pct', 'variable'] = 'Angry'
df1.loc[df1.variable=='fear_pct', 'variable'] = 'Fear'
df1.loc[df1.variable=='surprise_pct', 'variable'] = 'Surprise'
df1.loc[df1.variable=='disgust_pct', 'variable'] = 'Disgust'
df1.loc[df1.variable=='neutral_pct', 'variable'] = 'Neutral'

df1.value = np.round(df1.value*100,0)

fig = px.bar(df1, 
             x="name", 
             y="value",
             color='variable', 
             text='value',
             facet_col='variable',
             facet_row='source_bias',
             height=600, width=1000,
             labels=dict(name="", value="%", variable=""),
#             template="simple_white",
             title='What percent of the photos from news sources (by bias rating) contain facial expressions?<br>.  ' + timestamp_str,
             category_orders={"source_bias":['Left',
                                             'Lean Left',
                                             'Center',
                                             'Lean Right', 
                                             'Right'],
                             "variable":['Happy','Sad','Angry','Fear','Surprise','Disgust','Neutral']}
            )


# fig.update_xaxes(title_text='News Source Bias')
# fig.update_yaxes(title_text='')
fig.update_yaxes(range=[0,100])

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.show()
py.iplot(fig, filename='percent-by-source-bias-grid')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


emotion = 'happy'

row = face_df[emotion].idxmax()
url = face_df.loc[row,'urlToImage']
img_all = io.imread(url)
img_face = face_np[row]
prob = np.round(face_df.loc[row,emotion],2)
title = emotion + ": " + str(prob) + " probability"

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(url,title))

fig.add_trace(go.Image(z=img_all), 1, 1)
fig.add_trace(go.Image(z=img_face), 1, 2)

fig.update_xaxes(showticklabels=False, showgrid=False)
fig.update_yaxes(showticklabels=False, showgrid=False)
fig.update_layout(title_text= emotion + " - Image in latest query with highest predicted probability for this expression.   " + timestamp_str)


filename_str = 'show-image-with-highest-probability-' + emotion
# py.iplot(fig, filename=filename_str)
fig.show()



# COMMAND ----------


emotion = 'sad'

row = face_df[emotion].idxmax()
url = face_df.loc[row,'urlToImage']
img_all = io.imread(url)
img_face = face_np[row]
prob = np.round(face_df.loc[row,emotion],2)
title = emotion + ": " + str(prob) + " probability"

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(url,title))

fig.add_trace(go.Image(z=img_all), 1, 1)
fig.add_trace(go.Image(z=img_face), 1, 2)

fig.update_xaxes(showticklabels=False, showgrid=False)
fig.update_yaxes(showticklabels=False, showgrid=False)
fig.update_layout(title_text= emotion + " - Image in latest query with highest predicted probability for this expression.   " + timestamp_str)


filename_str = 'show-image-with-highest-probability-' + emotion
# py.iplot(fig, filename=filename_str)
fig.show()




# COMMAND ----------


emotion = 'angry'

row = face_df[emotion].idxmax()
url = face_df.loc[row,'urlToImage']
img_all = io.imread(url)
img_face = face_np[row]
prob = np.round(face_df.loc[row,emotion],2)
title = emotion + ": " + str(prob) + " probability"

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(url,title))

fig.add_trace(go.Image(z=img_all), 1, 1)
fig.add_trace(go.Image(z=img_face), 1, 2)

fig.update_xaxes(showticklabels=False, showgrid=False)
fig.update_yaxes(showticklabels=False, showgrid=False)
fig.update_layout(title_text= emotion + " - Image in latest query with highest predicted probability for this expression.   " + timestamp_str)


filename_str = 'show-image-with-highest-probability-' + emotion
# py.iplot(fig, filename=filename_str)
fig.show()



# COMMAND ----------

# Upload datafile to github repo

!pip install PyGithub
from github import Github

# ---------

git_file = 'fer_df.csv'
dbfs_file = '/dbfs/saved_df/fer_df.csv'

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
    