from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
sw = stopwords.words('english')


def get_wordnet_pos(nltk_tag):
    '''
    Translate NLTK part-of-speech tags to wordnet tags.
    
    Courtesy of Flatiron DS Live curriculum, lecture #64 "NLP Vectorization."
    '''
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
def parse_doc(doc, root = 'lemmatize', stop_words = sw, as_list = False):
    '''
    Modified from Flatiron DS Live curriculum, lecture #65 "NLP Modeling."
    
    :param doc: A string of text from the corpus.
    :param root: A string to determine whether the document will be...
                    lemmatized ('lemmatize'),
                    stemmed ('stem'),
                    or whether words will retain their full form.
    :param stop_words: Stopwords; defaults to NLTK's English stopwords list.
    :param as_list: Sets the object type for what the function returns (defaults to a string). If a list of tokens is preferable,
                    set as_list as True.
    :return: A document string in which all words have been...
                stripped of punctuation & numbers,
                made lowercase,
                parsed for stopwords,
                and lemmatized/stemmed.
    '''
    
    # Instantiate regular expression tokenizer
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    # Tokenize
    doc = regex_token.tokenize(doc)
    # Lowercase
    doc = [word.lower() for word in doc]
    # Remove stopwords
    doc = [word for word in doc if word not in stop_words]
    # Reducing to root words
    if root == 'lemmatize':
        lemmatizer = WordNetLemmatizer()
        doc = pos_tag(doc)
        doc = [(word[0], get_wordnet_pos(word[1])) for word in doc]
        doc = [lemmatizer.lemmatize(word[0], word[1]) for word in doc]
    elif root == 'stem':
        stemmer = PorterStemmer()
        doc = [stemmer.stem(word) for word in doc]
    else:
        pass
 
    # Output
    if as_list == True:
        return doc
    else:
        return ' '.join(doc)

    
def freq_out(df, col, n, stop_words = sw):
    '''
    Quick frequency distribution of the top n words from a document.
    
    :param df: DataFrame object
    :param col: Column from DataFrame on which to run the function.
    :param n: Number of most common items.
    :param stop_words: Stop words for parsing.
    :return: List of tuples containing most common words and number of occurrences.
    '''
    
    word_freq = FreqDist()
    
    for text in df[col].map(lambda x: parse_doc(x)):
        for word in text.split():
            word_freq[word] +=1
    return word_freq.most_commmon(n=n)