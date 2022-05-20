# Predicting Wikipedia Article Quality Using Natural Language Processing

![img](images/tomes.jpg)

*(photo courtesy of Dmitrij Paskevic, hosted on [Unsplash](https://unsplash.com/photos/YjVa-F9P9kk))*

### Authors
- **Luke Dowker** ([GitHub](https://github.com/toastdeini) | [LinkedIn](https://www.linkedin.com/in/luke-dowker/) | [Email](mailto:lhdowker@gmail.com))

# Overview

Over the course of its twenty-plus-year existence, Wikipedia's reputation has gradually evolved from that of a [digital "Wild West"](https://www.cnn.com/2009/TECH/08/26/wikipedia.editors/index.html) replete with [misinformation](https://usatoday30.usatoday.com/news/opinion/editorials/2005-11-29-wikipedia-edit_x.htm) to that of a [meticulously](https://en.wikipedia.org/wiki/Vandalism_on_Wikipedia#Prevention) curated and (generally) reliable resource for [fact-checking](https://en.wikipedia.org/wiki/Wikipedia_and_fact-checking) & bird's-eye/survey-level research. 

# Business Problem

Create a tool/model/application with natural language processing (NLP) that can predict whether a body of text, e.g. a Wikipedia article, meets objective standards of quality or if it is marked by a promotional tone, indicating potential for bias. 

# Data

Data used in this project is freely available for download on [Kaggle](https://www.kaggle.com/datasets/urbanbricks/wikipedia-promotional-articles), courtesy of user `urbanbricks`. "[Good articles](https://en.wikipedia.org/wiki/Wikipedia:Good_articles)" - articles which meet a "core set of editorial standards" - were stored as strings (with corresponding URLs) in one CSV file, `good.csv`. Articles with a "[promotional tone](https://en.wikipedia.org/wiki/Category:Articles_with_a_promotional_tone)" were stored in a separate CSV (`promotional.csv`) that, in addition to `text` and `url` columns, contains one-hot encoded columns that identify a subclass of promotional tone, e.g. `advert` (written like an advertisement) or `coi` (conflict of interest with subject).

![img](images/promo_dist.png)

It is important to note that the classes in discussion here - that is, whether an article meets the criteria for a "good article" or whether its contents are "promotional"/non-neutral - were **evaluated and labeled** by Wikipedia users and editors, and that this dataset (and consequently, these two classes) ***do not*** represent the full corpus of English-language Wikipedia.

A brief inquiry into the *length* of the documents belonging to each class revealed that `good` articles are, on average, about **three times longer** than `promotional` articles. This is, of course, inferential & descriptive rather than predictive, but it helps deepen our understanding of the data.

![img](images/avg_word_count.png)

# Methods

Initial exploration & analysis of the data utilized the [pandas](https://pandas.pydata.org/docs/index.html#) library for Python; exploratory visualizations were created using [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/). Preprocessing the data required modules from both [scikit-learn](https://scikit-learn.org/stable/) and NLTK ([Natural Language Toolkit](https://www.nltk.org/index.html)).

# Results

The performance baseline for analysis was an accuracy rate of **56%**, or `0.56` - 

<!-- Visualization of error -->

# Conclusions

- **Recommendation:** Justification
- **Recommendation:** Justification
- **Recommendation:** Justification

## Next Steps

- **Next step:** Justification
- **Next step:** Justification
- **Next step:** Justification

# Repository Structure
```
├── data
├── images
├── src
├── README.md
├── presentation.pdf
└── Final_Notebook.ipynb
```
## Further Reading and Citations
- Link to [Jupyter notebook](Final_Notebook.ipynb)
- Link to [non-technical presentation](presentation.pdf)