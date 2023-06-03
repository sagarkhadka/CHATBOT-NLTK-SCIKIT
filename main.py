import nltk
import numpy as np
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# opens text file - reads it - coverst to lower
f = open('corpus.txt', 'r', errors='ignore')
raw = f.read()

raw = raw.lower()

nltk.download('punkt')
nltk.download('wordnet')

sentence_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # coverst to list of word

# print(sentence_tokens)
# print(word_tokens)

# raw text pre-processing
lemmer = nltk.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# keyword matching
GREETING_INPUTS = ('hello', 'greetings', 'sup', 'hey')
GREETING_RESPONSES = ['Hi', 'Hey', '👋', 'Hi there', 'Hello']


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# generating response


def response(user_response):
    robo_response = ''
    sentence_tokens.append(user_response)

    # TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if (req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you."
        return robo_response
    else:
        robo_response = robo_response + sentence_tokens[idx]
        return robo_response


flag = True
print("BOT: Hello 👋. how are you doing.")

while (flag == True):
    user_response = input()
    user_response = user_response.lower()
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("BOT: You are welcome 😊")

        else:
            if (greeting(user_response) != None):
                print(f'BOT: {greeting(user_response)}')
                print('--------------------------------------')

            else:
                print('BOT: ', end='')
                print(response(user_response))
                sentence_tokens.remove(user_response)
                print('--------------------------------------')

    else:
        flag = False
        print('BOT: Bye! take care...')
