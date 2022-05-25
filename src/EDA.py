import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def word_count(document):
    '''
    Simple function to count the number of words in a string, passed in
    as `document`.
    '''
    
    # Split the document on whitespace to create a list of words
    words = document.split()
    # Check the length of the words list
    total = len(words)
    
    return total


def character_count(document):
    '''
    Simple function to count the number of characters in a string.
    '''
    
    # Remove whitespace
    new_doc = document.replace(' ', '')
    # Check the length of the string
    total = len(new_doc)
    
    return total



def compare_avg_counts(df, text_col, label_col, false_label, true_label, summary=True, create_plot=False):
    '''
    Used within the context of this project to compare word & character counts
    between `good` (label = 0) articles and `promotional` (label = 1) articles.
    
    Compares average word & character count across two labels in a dataframe.
    Returns a tuple of values.
    
    :param df: Dataframe object containing text and labels.
    :param text_col: String, name of column containing the words/characters to be counted.
    :param label_col: String, name of column containing the labels.
    :param false_label: int, string, or bool
    :param true_label: int, string, or bool
    '''

    avg_char_false = df.loc[df[label_col] == false_label][text_col].str.len().mean()
    avg_char_true = df.loc[df[label_col] == true_label][text_col].str.len().mean()
    
    split_words_false = df.loc[df[label_col] == false_label][text_col].str.split()
    split_words_true = df.loc[df[label_col] == true_label][text_col].str.split()
    
    word_count_false = 0
    word_count_true = 0
    
    for doc in split_words_false:
        word_count_false += len(doc)
        
    for doc in split_words_true:
        word_count_true += len(doc)
        
    avg_words_false = word_count_false / len(split_words_false)
    avg_words_true = word_count_true / len(split_words_true)
    
    if summary:
        print(f"Average document length, label {false_label}: {avg_words_false:.0f} words, {avg_char_false:.0f} characters.")
        print(f"Average document length, label {true_label}: {avg_words_true:.0f} words, {avg_char_true:.0f} characters.")
    
    if create_plot:
        plot_word_counts(avg_words_false, avg_words_true)

    return (avg_words_false, avg_words_true)
    
    
    
def plot_word_counts(f_average, t_average, false_label=0, true_label=1):
    '''
    
    Displays average word counts as a
    simple horizontal bar chart.
    
    Parameters
    ----------
    f_average : int or float
    t_average : int or float
        
    false_label : int or string, optional
    true_label : int or string, optional
        
        
    Returns
    -------
    Matplotlib figure displaying the average word
    counts across two classes.
    
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', which='major', pad=10)
    ax.tick_params(axis='x', which='major', pad=7)
    
    ax.barh(y=[f"Label {true_label}", f"Label {false_label}"],
            width=(t_average, f_average),
            color=['tab:red', 'tab:green'])
            
    ax.set_title("Average word count",
                 pad=15,
                 fontsize=28,
                 fontweight='bold')
    ax.set_xlabel("# of words",
                  labelpad=20,
                  fontsize=16)
    
    

def plot_subclass_dist(df, ):
    '''
    
    Parameters
    ----------
    df : pandas DataFrame object
        The dataframe containing the subclasses you'd
        like to count.
    
    
    Returns
    -------
    Matplotlib figure displaying the distribution
    of the desired subclasses.
    
    '''