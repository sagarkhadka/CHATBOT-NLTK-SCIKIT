import nltk
import numpy as np
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# opens text file - reads it - covers to lower
f = open('corpus.txt', 'r', errors='ignore')
raw = f.read()

raw = raw.lower()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

sentence_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # covers to list of words
stop_words = set(stopwords.words('english'))

# print(sentence_tokens)
# print(word_tokens)

# raw text pre-processing
lemmer = nltk.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    tokens = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
    tokens = [lemmer.lemmatize(token)
              for token in tokens if token not in stop_words]
    return tokens


# keyword matching
GREETING_INPUTS = ('hello', 'greetings', 'sup', 'hey')
GREETING_RESPONSES = ['Hi', 'Hey', '👋', 'Hi there', 'Hello']


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# generating response


# def response(user_response):
#     robo_response = ''
#     sentence_tokens.append(user_response)
#
#     TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
#     tfidf = TfidfVec.fit_transform(sentence_tokens)
#     vals = cosine_similarity(tfidf[-1], tfidf)
#     idx = vals.argsort()[0][-2]
#     flat = vals.flatten()
#     flat.sort()
#     req_tfidf = flat[-2]
#
#     if req_tfidf == 0:
#         robo_response = robo_response + "I am sorry! I don't understand you."
#     else:
#         # Filter out section headers (lines starting with '##')
#         content_sentences = [
#             sent for sent in sentence_tokens if not sent.startswith('##')]
#         robo_response = content_sentences[idx]
#
#     sentence_tokens.remove(user_response)
#     return robo_response

def response(user_response):
    robo_response = ''
    sentence_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    scores = list(enumerate(vals.flatten()))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    for idx, score in scores:
        if not sentence_tokens[idx].startswith('##'):
            robo_response = sentence_tokens[idx]
            break

    sentence_tokens.remove(user_response)
    return robo_response


flag = True
print("BOT: Hello 👋. how are you doing.")

while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("BOT: You are welcome 😊")
        else:
            if greeting(user_response) is not None:
                print(f'BOT: {greeting(user_response)}')
                print('--------------------------------------')
            else:
                print('BOT: ', end='')
                print(response(user_response))
                if user_response in sentence_tokens:
                    sentence_tokens.remove(user_response)
                print('--------------------------------------')
    else:
        flag = False
        print('BOT: Bye! take care...')
