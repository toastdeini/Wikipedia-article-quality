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
    
    
def parse_doc(doc, root = 'lemma', stop_words = sw, as_list = False):
    '''
    Modified from Flatiron DS Live curriculum,
    lecture #65 "NLP Modeling."
    
    Parameters
    ----------
    doc : string
        A string of text from the corpus.
    root : string, optional
        Determines whether the document will be...
        lemmatized ('lemma' - default),
        stemmed ('stem'),
        or unmodified (None/False)
    stop_words : list, optional
        Stopwords for removal. Defaults to NLTK's
        English stopwords list.
    as_list : bool, optional
        Sets object type for function's returned
        object. Defaults to a string, i.e. `as_list
        = False`.
    
    Returns
    -------
    doc
        a string or list (see parameter `as_list`), in
        which all words have been...
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
    if root == 'lemma':
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

test_article_string = "**THIS IS SAMPLE TEXT**\n\nThe Papuan mountain pigeon\
 (Gymnophaps albertisii) is a species of bird in the pigeon family Columbidae.\
 It is found in the Bacan Islands, New Guinea, the D'Entrecasteaux Islands,\
 and the Bismarck Archipelago, where it inhabits primary forest,\
 montane forest, and lowlands. It is a medium-sized species of pigeon,\
 being 33–36 cm (13–14 in) long and weighing 259 g (9.1 oz) on average.\
 Adult males have slate-grey upperparts, chestnut-maroon throats and bellies,\
 whitish breasts, and a pale grey terminal tail band. The lores and orbital region\
 are bright red. Females are similar, but have grayish breasts and grey edges to\
 the throat feathers.\n\nThe Papuan mountain pigeon is frugivorous, feeding on\
 figs and drupes. It breeds from October to March in the Schrader Range, but\
 may breed throughout the year across its range. It builds nests out of sticks\
 and twigs in a tree or makes a ground nest in short dry grass, and lays a single egg.\
 The species is very social and is usually seen in flocks of 10–40 birds, although\
 some groups can have as many as 80 individuals. It is listed as being of least concern\
 by the International Union for Conservation of Nature (IUCN) on the IUCN Red List due\
 to its large range and lack of significant population decline.\n\nThe Papuan mountain\
 pigeon was described as Gymnophaps albertisii by the Italian zoologist Tommaso Salvadori\
 in 1874 on the basis of specimens from Andai, New Guinea. It is the type species of the\
 genus Gymnophaps, which was created for it.[3] The generic name is derived from the\
 Ancient Greek words γυμνος (gumnos), meaning bare, and φαψ (phaps), meaning pigeon.\
 The specific name albertisii is in honour of Luigi D'Albertis, an Italian botanist and\
 zoologist who worked in the East Indies and New Guinea.[4] Papuan mountain pigeon is\
 the official common name designated by the International Ornithologists' Union.\
 Other common names for the species include mountain pigeon (which is also used for\
 Gymnophaps pigeons in general), bare-eyed mountain pigeon, bare-eyed pigeon (which is\
 also used for Patagioenas corensis), and D'Albertis's mountain pigeon.\n\nThe Papuan\
 mountain pigeon is one of four species in the mountain pigeon genus Gymnophaps in the pigeon\
 family Columbidae, which is found in Melanesia and the Maluku Islands. It forms a\
 superspecies with the other species in its genus. Within its family, the genus\
 Gymnophaps is sister to Lopholaimus, and these two together form a clade sister to\
 Hemiphaga.[8] The Papuan mountain pigeon has two subspecies:[a][3]"


# Commented out due to expensive computation time - needs reworked

# def freq_out(df, col, n, stop_words = sw):
#     '''
#     Quick frequency distribution of the top n words from a document.
    
#     :param df: DataFrame object
#     :param col: Column from DataFrame on which to run the function.
#     :param n: Number of most common items.
#     :param stop_words: Stop words for parsing.
#     :return: List of tuples containing most common words and number of occurrences.
#     '''
    
#     word_freq = FreqDist()
    
#     for text in df[col].map(lambda x: parse_doc(x)):
#         for word in text.split():
#             word_freq[word] +=1
#     return word_freq.most_commmon(n=n)