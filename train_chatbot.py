import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer() #Lematizer is like a dictionary that maps words to their root form
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json', encoding="utf8").read()
intents = json.loads(data_file)
print(intents)      # intents: grupos de conversaciones típicas
                    # patterns: posibles interacciones de los usuarios

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenizar cada palabra
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # añadir a los documentos de la matriz
        documents.append((w, intent['tag']))

        # añadiendo clases a nuestra lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

# preparación para la formación en red
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # bag of words
    bag = []
    # lista de tokens
    pattern_words = doc[0]
    # lemmatizzazione de token
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # si la palabra coincide, introduzco 1, en caso contrario 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])


training = np.array(training)    
# Creación de conjuntos de entrenamiento y prueba: X - patrones, Y - intenciones
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Trainning data created")

# creacion del modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#ajustar y guardar el modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=900, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Modelo creado")