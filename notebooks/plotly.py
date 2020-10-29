# Databricks notebook source
import numpy as np
import pandas as pd

!pip install chart_studio
!pip install plotly --upgrade

import chart_studio
import chart_studio.plotly as py
from chart_studio.plotly import plot, iplot
import plotly.graph_objects as go
import plotly.express as px


chart_studio.tools.set_credentials_file(username='adamjd', 
                                        api_key='1piYaG7ohT90yKxLEjL7')




# COMMAND ----------


import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import csv

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


# SAVED AS percent-by-source-bias-grid


df1 = fer_df.groupby(['source_bias','name'], as_index=False).sum()
df1['happy_pct'] = df1.top_happy/df1.emotions
df1['sad_pct'] = df1.top_sad/df1.emotions
df1['angry_pct'] = df1.top_angry/df1.emotions
# df1['fear_pct'] = df1.top_fear/df1.emotions
# df1['surprise_pct'] = df1.top_surprise/df1.emotions
# df1['disgust_pct'] = df1.top_disgust/df1.emotions
df1['neutral_pct'] = df1.neutral/df1.emotions


df1 = pd.melt(df1, 
              id_vars=['source_bias','name'],
              value_vars=['happy_pct', 
                   'sad_pct', 
                   'angry_pct', 
#                    'fear_pct', 
#                    'surprise_pct', 
#                    'disgust_pct', 
                   'neutral_pct'])

df1 = df1.groupby(['source_bias','name','variable'], as_index=False).mean()

df1.sort_values(['name'], ascending=False, inplace=True)

df1.loc[df1.variable=='happy_pct', 'variable'] = 'Happy'
df1.loc[df1.variable=='angry_pct', 'variable'] = 'Angry'
df1.loc[df1.variable=='neutral_pct', 'variable'] = 'Neutral'
df1.loc[df1.variable=='sad_pct', 'variable'] = 'Sad'

df1.value = np.round(df1.value*100,0)

fig = px.bar(df1, 
             x="name", 
             y="value",
             color='name', 
             facet_col='source_bias',
             facet_row='variable',
             height=600, width=1000,
             labels=dict(name="", value="%"),
#             template="simple_white",
             title='What percent of the photos from news sources with each bias rating show <br>Joe Biden and Donald Trump with either Happy, Sad, Angry, or Neutral facial expressions?',
             category_orders={"source_bias":['Left',
                                             'Lean Left',
                                             'Center',
                                             'Lean Right', 
                                             'Right'],
                             "variable":['Happy','Sad','Angry','Neutral']}
            )


# fig.update_xaxes(title_text='News Source Bias')
# fig.update_yaxes(title_text='')
fig.update_yaxes(range=[0,100])

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.show()
py.iplot(fig, filename='percent-by-source-bias-grid')

# COMMAND ----------



fig = px.bar(fer_df, 
             x="date", 
             color='name', 
             facet_col='source_bias',
             height=600, width=1400,
             labels=dict(name="", value="%"),
#              template="simple_white",
             title='Number of news articles in search query containing images of Donald Trump and Joe Biden (from past week).',
             category_orders={"source_bias":['Left',
                                             'Lean Left',
                                             'Center',
                                             'Lean Right', 
                                             'Right']}
            )

 

fig.update_xaxes(tickformat="%m-%d", 
                 tickangle=-90,
                 title_text='',
                 type='category')

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.show()
py.iplot(fig, filename='articles-containing-trump-biden-photos-past-week')


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
             title="How many face images returned by the search query were classified for each facial expression?",
             labels=dict(name="", top_emotion_sum="Number of face images classified", top_emotion="Predicted Facial Expression"),     
#              template="simple_white",
             category_orders={"top_emotion":['Happy','Sad','Angry','Fear','Surprise','Disgust','Neutral']}
            )

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.show()
py.iplot(fig, filename='facial-expression-totals-barchart')



# COMMAND ----------


df1 = fer_df
df1.sort_values(['name'], ascending=False, inplace=True)

fig = px.bar(df1, 
             y='source',
             x='num_articles',             
             title='How many total news articles were collected from each news source?',
             labels=dict(name="", source="News Source", count="Total Articles Collected in Latest Query"),     
             height=600, width=600)

fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()

py.iplot(fig, filename='total-articles-by-source')

# COMMAND ----------


df1 = fer_df
df1.sort_values(['name'], ascending=False, inplace=True)

fig = px.bar(df1, 
             y='source',
             color='name',
             title='How many images collected from each news source contained faces of<br> Donald Trump and Joe Biden, according to the model?',
             labels=dict(name="", source="News Source", count="Number of Faces Detected in Collected Articles"),     
             height=600, width=700)

fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()
py.iplot(fig, filename='total-faces-detected-by-source')

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
             title='What were the distributions of probabilities returned by the facial recognition<br>model for each type of facial expression?',
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
             title='For the highest probability expression for each face image (the "top emotion"),<br> what were the distributions of those probabilities?',
             labels=dict(top_emotion_prob="Probabilities of Top Expression for Each Face", top_emotion="Top Emotion"),
             category_orders={"top_emotion":['Happy','Sad','Angry','Fear','Surprise','Disgust','Neutral']})


fig.update_yaxes(range=[0,1])

fig.show()
py.iplot(fig, filename='top-emotion-probability-boxplots')


# COMMAND ----------

!pip install scikit-image

# COMMAND ----------

import skimage
from skimage import io

urls = []
for i in ['Happy','Sad','Angry','Fear','Surprise','Disgust','Neutral']:

    df1 = fer_df.loc[fer_df.top_emotion == i,:]
    df1 = df1.sort_values(['top_emotion_prob'], ascending=False)
    df1 = df1.reset_index(drop=True)
    print(df1.loc[0,'urlToImage'])
    urls.append(df1.loc[0,'urlToImage'])



fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.imshow(io.imread(urls[0])),
    row=1, col=1
)

fig.add_trace(
    go.imshow(io.imread(urls[1])),
    row=1, col=2
)

fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
fig.show()

# COMMAND ----------

face_df

# COMMAND ----------


from plotly.subplots import make_subplots
import skimage
from skimage import io

fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3])


row = face_df['sad'].idxmax()
img_face = face_np[row]
url = face_df.loc[row,'urlToImage']
img_all = io.imread(url)
fig.add_trace(go.Image(z=img_all), 1, 1)
fig.add_trace(go.Image(z=img_face), 1, 2)
fig.show()

# row = face_df['sad'].idxmax()
# img = face_np[row]
# fig.add_trace(go.Image(z=img), 1, 2)

# row = face_df['angry'].idxmax()
# img = face_np[row]
# fig.add_trace(go.Image(z=img), 1, 3)

# row = face_df['fear'].idxmax()
# img = face_np[row]
# fig.add_trace(go.Image(z=img), 1, 4)

# row = face_df['surprise'].idxmax()
# img = face_np[row]
# fig.add_trace(go.Image(z=img), 2, 1)

# row = face_df['disgust'].idxmax()
# img = face_np[row]
# fig.add_trace(go.Image(z=img), 2, 2)

# row = face_df['neutral'].idxmax()
# img = face_np[row]
# fig.add_trace(go.Image(z=img), 2, 3)


# fig.show()    
