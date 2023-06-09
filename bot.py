import constants
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import warnings


def get_response(res):
    if any(remove_punctuation(res).lower() in remove_punctuation(greet).lower() for greet in constants.GREETING_INPUTS):
        return random.choice(constants.GREETING_REPLIES)
    elif any(remove_punctuation(res).lower() in remove_punctuation(thanks).lower() for thanks in constants.THANKS_INPUTS):
        return random.choice(constants.THANKS_REPLIES)


def get_lemmatized_tokens(text):
    normalized_tokens = nltk.word_tokenize(
        remove_punctuation(text.lower()))
    return [nltk.stem.WordNetLemmatizer().lemmatize(normalized_token) for normalized_token in normalized_tokens]


def get_query_reply(query):
    documents.append(query)
    tfidf_results = TfidfVectorizer(
        tokenizer=get_lemmatized_tokens, stop_words='english').fit_transform(documents)
    cosine_similarity_results = cosine_similarity(
        tfidf_results[-1], tfidf_results).flatten()
    # The last will be 1.0 because it is the Cosine Similarity between the first document and itself
    best_index = cosine_similarity_results.argsort()[-2]
    documents.remove(query)
    if cosine_similarity_results[best_index] == 0:
        return "I am sorry! I don't understand you..."
    else:
        return documents[best_index]


def remove_punctuation(text):
    punctuation_marks = dict((ord(punctuation_mark), None)
                             for punctuation_mark in string.punctuation)
    return text.translate(punctuation_marks)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    corpus = open('corpus.txt', 'r', errors='ignore').read().lower()
    documents = nltk.sent_tokenize(corpus)

    print('Bot: I will answer your queries about Chat bots. If you want to exit just type: Bye!')
    print('--------------------------')
    end_chat = False
    while end_chat == False:
        input_text = input()
        if remove_punctuation(input_text).lower() != 'bye':
            formality_reply = get_response(input_text)
            if formality_reply:
                print('Bot: ' + formality_reply)
                print('--------------------------')
            else:
                print('Bot: ' + get_query_reply(input_text))
                print('--------------------------')
        else:
            print('Bot: Bye! Take care ' +
                  random.choice(constants.CANDIES))
            end_chat = True
