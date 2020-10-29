# Databricks notebook source
import logging 
logging.getLogger("py4j").setLevel(logging.ERROR)


# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

!pip install opencv-python
import cv2

!pip install requests
!pip install urllib

!pip install cmake
!pip install boost
!pip install boost-python --with-python3

!pip install dlib

!pip install face_recognition
!pip install fer

import requests
import urllib
from urllib.parse import urlparse
import dlib
import face_recognition
from fer import FER
import datetime


opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


spark

# COMMAND ----------

face_df = pd.read_csv('/dbfs/saved_df/face_df.csv', header=0, index_col=0)

# COMMAND ----------

face_np = np.load('/dbfs/faces/face_np.npy', allow_pickle=True)

# COMMAND ----------


detector = FER(mtcnn=True)

for i,x in face_df.iterrows():
    
    face = face_np[i]
    detect_dict = detector.detect_emotions(face)
    
#     plt.imshow(face)
#     title = str(detect_dict)
#     plt.title(title)
#     plt.show()
        
    if (bool(detect_dict)):
        face_df.loc[i,'emotions'] = True
        face_df.loc[i,'happy'] = detect_dict[0]['emotions']['happy']
        face_df.loc[i,'sad'] = detect_dict[0]['emotions']['sad']
        face_df.loc[i,'angry'] = detect_dict[0]['emotions']['angry']
        face_df.loc[i,'fear'] = detect_dict[0]['emotions']['fear']
        face_df.loc[i,'surprise'] = detect_dict[0]['emotions']['surprise']
        face_df.loc[i,'disgust'] = detect_dict[0]['emotions']['disgust']
        face_df.loc[i,'neutral'] = detect_dict[0]['emotions']['neutral']
    else:
        face_df.loc[i,'emotions'] = False



# COMMAND ----------

face_df.loc[face_df.emotions==False,:].index

# COMMAND ----------

face_df.to_csv('/dbfs/saved_df/face_df.csv')

# COMMAND ----------

# Upload datafile to github repo

!pip install PyGithub
from github import Github

# ---------

git_file = 'face_df.csv'
dbfs_file = '/dbfs/saved_df/face_df.csv'

# ---------

f = open("/dbfs/github_token.txt", "r")
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
    