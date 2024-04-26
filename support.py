import nltk 
from nltk.stem import WordNetLemmatizer
import pickle
import json
words = []
classes = []
intents = json.loads(open('intents.json').read())
documents = []
ignore_letters = ['!', '?', ',', '.']
lemmatizer = WordNetLemmatizer()
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
