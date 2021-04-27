from flask import Flask, render_template, url_for, request, redirect
import pickle
import numpy as np
import requests
# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
from tensorflow.keras.models import load_model
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
#create chatbot
# englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
# trainer = ChatterBotCorpusTrainer(englishBot)
# trainer.train("chatterbot.corpus.english") #train the chatter bot for english
model = load_model('chatbot.h5')
intents = json.loads(open('intents.json').read())
classes = pickle.load(open('classes.pkl','rb'))
words = pickle.load(open('words.pkl','rb'))
#define app routes

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
  return sentence_words 

def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  bag = [0] * len(words)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word == w :
        bag[i] = 1
  return np.array(bag)

def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD = 0.20
  results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD ]

  results.sort(key=lambda x : x[1],reverse = True)
  return_list = []
  for r in results : 
    return_list.append({'intent':classes[r[0]] , 'probablity' : str(r[1])})
  return return_list

def get_response(intents_list,intents_json):
  tag = intents_list[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents : 
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      break
  return result


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
  userText = request.args.get('msg')
  print(userText)
  userText = str(userText)
  print(userText)
  ints = predict_class(userText)
  res = get_response(ints, intents)
  print(res,userText)
  return str(res)

#    return str(englishBot.get_response(userText))

if __name__ == "__main__":
	print(type(classes))
	app.run(debug=True)