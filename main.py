import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from collections import Counter 
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings 
import re 
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


df = pd.read_csv('reddit_hate_speech.csv')

print("Dataset shape: ", df.shape)
print("\nFirst few rows: ")
print(df.head())

print("\nColumn names: ")
print(df.columns)

print("\nData types: ")
print(df.dtypes)

print("\Class distribution: ")
print(df['class'].value_counts())


def clean_text(text):
    """
    Clean text by converting to lowercase and remove special characters
    """
    text = str(text)

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = " ".join(text.split())

    return text

def get_tokens(text):
    """
    Tokenize text into words
    """
    tokens = word_tokenize(text)

    tokens = [token for token in tokens if token.isalpha()]

    return tokens

def count_pronouns(tokens):
    """
    Count pronouns in a list of tokens using POS tagging
    """

    # Returns list of tuples: [('I', 'PRP'), ('love', 'VB'), ...]
    pos_tags = nltk.pos_tag(tokens)

    pronoun_count = sum(1 for word, tag in pos_tags if tag in ['PRP', 'PRP$'])

    return pronoun_count

text = "I love my dog"
tokens = get_tokens(text)
print(f"Total pronouns: {count_pronouns(tokens)}")



