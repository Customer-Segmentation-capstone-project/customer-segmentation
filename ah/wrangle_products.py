import pandas as pd
import re
import os
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import unicodedata

# ==========================================================================

def acquire_amazon():
    '''
    This will acqurie the Cycling category Amazon product data from either local cached
    file or from kaggle.com
    
    returns: uncleaned dataframe
    '''
    # set filename
    filename = 'amazon_cycling.csv'
    # check if local cached version of the file exists
    if os.path.exists(filename):
        # display status message
        print(f'Opening local {filename} file')
        # open local file data
        df = pd.read_csv(filename)
    # if there is no local file
    else:
        # display status message
        print(f'Local file {filename} not found')
        print('downloading data')
        # set url path to the dataset
        path = '''https://storage.googleapis.com/kagglesdsdata/datasets/3020336/5239462/Cycling.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230601%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230601T202559Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=609985cbe62280925ce349da43238b08dbf97feeee9055c7cdb98617052093e28ff6a4060b615d16f53fd2246ae70e0baeb950a2770e327998a184339d881b88bc0cfbfb2bde36af0544af4ada38bbff28f84957b8f48cc9a1bad4e15bba19190b9b992d475d4e80f8568dfd8d95b6b9c89bb60e8f75eaf79068e4ad36bab6c9bd69971f0c6d5b101e72684b407b88490c1471ff4a94540668165830c302eb3128389382028d84b6b438901e81f51a61c67e9dd6da74c0d4f2028582533573c808ab1218a5924a2d071bad89171fbaf634ce225b68775a0f193ea8c3230e19dce835467a56ac894db017586defae68bc8c175d3655edfcd73997e635c77932a2'''
        # read the data from the url path
        df = pd.read_csv(path)
        # cache the data to local csv file
        df.to_csv(filename, index=False)
    # return the dataframe
    return df

# ===================================================================================

def basic_clean(original_string):
    '''
    This will take in a string, make it all lowercase, normalize the characters to ascii
    and remove characters that are not letters, numbers or spaces
    '''
    # normalize the characters to ascii standart
    normalized = unicodedata.normalize('NFKD', original_string).\
        encode('ascii', 'ignore').decode('utf-8')
    # lowercase all the words in the data
    lowered = normalized.lower()
    # remove things that arent letters, numbers and spaces
    basic_cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', lowered)
    #return the cleaned string
    return basic_cleaned

# ===================================================================================

def tokenize(basic_cleaned):
    '''
    This will break up words into smaller, discrete (tokenized) units
    '''
    # grab our tokenizer from nltk
    tokenizer = nltk.tokenize.ToktokTokenizer()
    # tokenize the data
    tokenized = tokenizer.tokenize(basic_cleaned, return_str=True)
    # return the tokenized data
    return tokenized

# ===================================================================================

def lemmatize(tokenized):
    '''
    This will cut a string of words down into their root words (Lemmatizing)
    '''
    # create lemmatizer object
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # lemmatize every word in the string
    lemmatized = ' '.join([lemmatizer.lemmatize(word) for word in tokenized.split()])
    # return the lemmatized string
    return lemmatized

# ===================================================================================

def remove_stopwords(string, extra_words=None, exclude_words=None):
    '''
    This will remove words that hold little meaning to a machine learning system
    such as: 'the' 'am', 'is', 'are',
    '''
    # get a list of the stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    # add extra words to the stopwords list
    if extra_words:
        [stopwords.append(word) for word in extra_words]
    # remove the exclude words from the stopwords list if word is in the stopwords list
    if exclude_words:
        [stopwords.remove(word) for word in exclude_words 
                     if (word in stopwords)]
    # get the list of words that are not in the stopwords
    stops_removed = ' '.join([word for word in string.split() 
                              if word not in stopwords])
    # return the words not in the stopwords list
    return stops_removed

# ===================================================================================

def prepare_amazon(df, extra_words=None, exclude_words=None):
    '''
    This will clean/normalize, tokenize and lemmatize the product names from amazon in 
    preparation of using nlp on it.
    '''
    # create an empty list to store the names
    names = []
    # cycle through all the product names
    for name in df.name:
        # clean the product name
        basic_cleaned = basic_clean(name)
        # tokenize the words in the name
        tokenized = tokenize(basic_cleaned)
        # remove the stopwords from the product name
        cleaned = remove_stopwords(tokenized, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)
        # get the lemmatized words from the product name
        lemmatized = lemmatize(cleaned)
        # add the lemmatized name to the list of names
        names.append(lemmatized)
        
    # add the lemmatized version of the name to the original df
    df['name_preped'] = names
    # return the df
    return df