import numpy as np
import pandas as pd
import json
import random
import pickle
import nltk
import re
import requests

from tensorflow.keras.preprocessing.sequence import pad_sequences
import sklearn
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from keras.models import load_model
from fastapi import FastAPI, Request, WebSocket, WebSocketException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model = load_model("chatbot_model.h5")
sia = SentimentIntensityAnalyzer()
with open('intents.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])
dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
        
df = pd.DataFrame.from_dict(dic)
tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])
texts = []
msg = list()
text = list()
result = {"track"}

def song_emotion():
    compiled_text = ""
    for t in texts[1:-1]: 
        compiled_text += " " + t
    
    pol_sc = sia.polarity_scores(compiled_text)
    highest_rate = max(pol_sc.values())
    em = [i for i in pol_sc.keys() if pol_sc[i] == highest_rate]
    return em


def generate_answer(pattern): 
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)
        
    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test).squeeze()
    print([x_test])
    x_test = pad_sequences([x_test], padding='post', maxlen=18)
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})

@app.websocket("/ws")
async def chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        texts.append(data)
        print(data)
        if data == "Hello server":
            await websocket.send_text("""Hey there, I am Alex! Tell me how you are feeling today and I will help you find a song you'd like to hear right now.\n
            If you'd like me to tell you a song now, type \"Give me some songs\" and hit send. If you'd like to stop chatting, just type \"END\" and hit the send button.""")
        elif data.upper == "END":
            await websocket.send_text("Okay, it's been great talking to you! Bye!")
            await websocket.close()
        elif data == "Give me some songs":
            await websocket.send_text("Please wait, fetching some songs for you....")
            em = song_emotion()
            if (em == ['neg']):
                mytag = "sad"
            elif (em == ["pos"]):
                mytag = "happy"
            elif (em == ["neu"]):
                mytag = "relaxing"
            else:
                mytag = "lofi"
            
            print("Tag is:", mytag)
            
            url=f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={mytag}&api_key=19a48b20b3f6507595138d57f5af0eb7&format=json&limit=10"
            dict1 = dict()
            response = requests.get(url)
            result = response.json()["tracks"]
            recoms = ""
            for track in result["track"]: 
                # dict1[track["name"]] = track["url"]
                recoms += "Song name: " + track["name"]
                recoms += "\nSong url: " + track["url"] + "\n"
            await websocket.send_text(recoms)
            
        else:
            res = generate_answer(data)
            await websocket.send_text(res)
