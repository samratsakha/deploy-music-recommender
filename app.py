# Required Libraries
from flask import Flask, render_template, request, make_response
import jsonify
import requests
import json
from requests.sessions import Request
import pickle
import numpy as np
import pandas as pd


# Load the recommender model
model = pickle.load(open('nn_recommender.pkl','rb'))


app = Flask(__name__)
        
## Load the dataset into a dataframe
df = pd.read_csv('dataset.csv')
df_cleaned = pd.read_csv('cleaned.csv')
df_url = pd.read_csv('final_url_data.csv')


track_names = list(df['Track Name'])
urls = list(df_url['URL'])


def recommend(distances,indices):

    music_list = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            pass
        else:
            temp = [df.iloc[indices.flatten()[i],0] , df.iloc[indices.flatten()[i],1] , df.iloc[indices.flatten()[i],2]]
            music_list.append(temp)

    music_list.sort(key= lambda X:X[2])

    return music_list


def recieve_url(indices):

    url_list = []
    
    for i in range(len(indices.flatten())):
        if i==0:
            pass
        else:
            temp = [urls[indices.flatten()[i]],df.iloc[indices.flatten()[i],2]]
            url_list.append(temp)

    url_list.sort(key= lambda X:X[1])

    final_url_list = []

    for i in range(len(url_list)):
        final_url_list.append(url_list[i][0])
    
    return final_url_list



# Templates
# Home page
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



# API JSON
@app.route("/to_model", methods=['POST'])
def to_model():

    req = request.get_json()
    track_n = req['val_array']

    try:

        index = ""

        for i in range(len(track_names)):
            if track_names[i]==track_n:
                index = i

        distances, indices = model.kneighbors(df_cleaned.iloc[index,:].values.reshape(1, -1), n_neighbors = 11)

        music_list = recommend(distances,indices)

        recommend_list = []

        selected_track = urls[indices.flatten()[0]]

        for i in range(len(music_list)):
            recommend_list.append(music_list[i][1]+ " by " +music_list[i][0])

        outs = recommend_list
        outs2 = recieve_url(indices)
        

        x = {"output": outs, "url": outs2, "track":selected_track}
        y = json.dumps(x)

        return y

    except:
        x = {"output": "ERROR"}
        y = json.dumps(x)

        return y


@app.route("/to_tracks", methods=['POST'])
def to_tracks():

    req = request.get_json()
    stock_name = req['track_array']

    try:

        outs = track_names

        x = {"output": outs}
        y = json.dumps(x)

        return y

    except:
        x = {"output": "ERROR"}
        y = json.dumps(x)

        return y



if __name__=="__main__":
    app.run(debug=True)

