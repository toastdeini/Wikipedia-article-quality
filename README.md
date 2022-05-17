# Wikipedia Article Quality

![img](images/tomes.jpg)

*(photo courtesy of Dmitrij Paskevic, hosted on [Unsplash](https://unsplash.com/photos/YjVa-F9P9kk))*

### Authors
- **Luke Dowker** ([GitHub](https://github.com/toastdeini) | [LinkedIn](https://www.linkedin.com/in/luke-dowker/) | [Email](mailto:lhdowker@gmail.com))

# Overview

Over the course of its twenty-plus-year existence, Wikipedia's reputation has gradually evolved from that of a digital ["Wild West"](https://www.cnn.com/2009/TECH/08/26/wikipedia.editors/index.html) replete with [misinformation](https://usatoday30.usatoday.com/news/opinion/editorials/2005-11-29-wikipedia-edit_x.htm) to that of a [meticulously](https://en.wikipedia.org/wiki/Vandalism_on_Wikipedia#Prevention) curated and (generally) reliable resource for [fact-checking](https://en.wikipedia.org/wiki/Wikipedia_and_fact-checking) & bird's-eye/survey-level research. 

# Business Problem

Create a tool/model/application with natural language processing (NLP) that can predict whether a body of text, e.g. a Wikipedia article, meets objective standards of quality or if it is marked by a promotional tone, indicating potential for bias. 

# Data

Data used in this project is freely available for download on [Kaggle](https://www.kaggle.com/datasets/urbanbricks/wikipedia-promotional-articles), courtesy of user `urbanbricks`. "[Good articles](https://en.wikipedia.org/wiki/Wikipedia:Good_articles)" - articles which meet a "core set of editorial standards" - were stored as strings (with corresponding URLs) in one CSV file, `good.csv`. Articles with a "[promotional tone](https://en.wikipedia.org/wiki/Category:Articles_with_a_promotional_tone)" were stored in a separate CSV (`promotional.csv`) that, in addition to `text` and `url` columns, contains one-hot encoded columns that identify a subtype of promotional tone, e.g. `advert` (written like an advertisement - this accounts for the majority of documents in the dataset) or `coi` (conflict of interest with subject).

<!-- image of imbalanced classes in promotional.csv dataset to justify pursuit of binary classification as starter problem -->

It is important to note that the classes in discussion here - that is, whether an article meets the criteria for a "good article" or whether its contents are "promotional"/non-neutral - were **evaluated and labeled** by Wikipedia users and editors. 

# Methods

- Relevant `scikit-learn` libraries
- Relevant `nltk` modules
    - `CountVectorizer` / `TfidfVectorizer`
- Description of iterative modeling process

# Results

- What algorithms yielded the best results? Why? What was the process like?
- What *are* those results? What can we conclude from them?

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