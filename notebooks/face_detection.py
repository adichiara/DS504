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

# make backups of latest data files

dbutils.fs.cp('/saved_df/face_df.csv','/archive/face_df.csv')
dbutils.fs.cp('/faces/face_np.npy','/archive/face_np.npy')


# COMMAND ----------

# delete existing image files to be replaced

# COMMAND ----------

# MAGIC %fs rm -r '/img_dir/' 

# COMMAND ----------

dbutils.fs.mkdirs('/img_dir/')

# COMMAND ----------

article_df = pd.read_csv("/dbfs/saved_df/article_df.csv", header=0, index_col=0)

# COMMAND ----------

# read images from URLs and save to image files. 

for i, x in article_df.iterrows():
    if (i % 100 == 0):
        print(i)
        
    filename = 'URL_img_' + str(i) + '.jpg'
    oldpath = 'file:/databricks/driver/' + filename
    movepath = '/img_dir/' + filename
    newpath = '/dbfs/img_dir/' + filename    

    if (x['urlToImage']):
        try: 
            urllib.request.urlretrieve(x['urlToImage'], filename)
            #move file from local path to dbfs
            dbutils.fs.mv(oldpath, movepath)
            article_df.loc[i,'image_path'] = newpath
            article_df.loc[i,'image_file'] = filename
        except:    
            print('could not load:',i,'of',article_df.shape[0])
            continue
    
# read images into Spark image dataframe
image_df = spark.read.format("image").load('/img_dir/')


# COMMAND ----------

# links to official portraits
trump_url = 'https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg'
biden_url = 'https://upload.wikimedia.org/wikipedia/commons/f/f4/Joe_Biden_official_portrait_2013.jpg'

detector = FER(mtcnn=True)

known_faces_df = pd.DataFrame({'name':['Donald Trump',
                                       'Joe Biden'],
                               'url':['https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg',
                                      'https://upload.wikimedia.org/wikipedia/commons/f/f4/Joe_Biden_official_portrait_2013.jpg']})


known_faces_df['face'] = np.NaN
known_faces_df['face'] = known_faces_df['face'].astype(object)
known_faces_df['encoding'] = np.NaN
known_faces_df['encoding'] = known_faces_df['encoding'].astype(object)

for i,x in known_faces_df.iterrows():

    urllib.request.urlretrieve(x['url'], 'URL_img.jpg')

    image = cv2.imread('URL_img.jpg')
    img = np.float32(image)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = detector.find_faces(img)    
    bx,by,bw,bh = boxes[0]
    x1 = bx
    x2 = bx+bw
    y1 = by
    y2 = by+bh

    face = img_rgb[y1:y2,x1:x2]

    plt.imshow(face)
    plt.show()

    encoding = face_recognition.face_encodings(face)[0]

    known_faces_df.at[i,'face'] = face
    known_faces_df.at[i,'encoding'] = encoding

# COMMAND ----------


face_list = []
face_df = pd.DataFrame()
face_np = np.array([])

for i,x in article_df.iterrows():
    if (i % 100 == 0):
        print(i)
    
    try:
        image = face_recognition.load_image_file(x['image_path'])
        
    except:
        continue
    
#     plt.imshow(image)
#     plt.show()

    try:
        face_locations = face_recognition.face_locations(image)
    except:
        continue

    for box in face_locations:
#         print(box)
        top, right, bottom, left = box
        face = image[top:bottom,left:right]
        
#         plt.imshow(face)
#         plt.show()

        face_encodings = face_recognition.face_encodings(face)

        if (face_encodings):
            matches = face_recognition.compare_faces(list(known_faces_df['encoding']),
                                                 face_encodings[0], 
                                                 tolerance=.6)

            if (sum(matches)>0):
#                 print(i,"match")
                name = known_faces_df.loc[matches,'name']
                name = name[name.index[0]]

#                 plt.imshow(face)
#                 plt.title(name)
#                 plt.show()
#                 face_list.append(face)

                face_dict = { 'name': name,
                            'source': x['source_name'],
                            'title': x['title'],
                            'urlToImage': x['urlToImage'],
                            'date': x['date'],
                            'year': x['year'],
                            'month': x['month'],
                            'day': x['day']}

                face_df = face_df.append(pd.Series(face_dict), ignore_index=True)
        
                face_list.append(face)

                              
face_np = np.array(face_list)

# t1 = datetime.now()
# print(t1-t0)




# COMMAND ----------

print(face_df.name.value_counts()['Joe Biden'])
print(face_df.name.value_counts()['Donald Trump'])
      

# COMMAND ----------

np.save('/dbfs/faces/face_np.npy', face_np)

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
    