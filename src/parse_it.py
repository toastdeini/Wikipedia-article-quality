from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
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
        doc = [ ( word[0] , get_wordnet_pos( word[1] ) ) for word in doc]
        doc = [lemmatizer.lemmatize( word[0], word[1] ) for word in doc]
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