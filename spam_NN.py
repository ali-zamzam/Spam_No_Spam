
from sklearn import preprocessing
import numpy as np
import pandas as pd
import os


def derivative(x):
    return x * (1.0 - x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


NEWLINE = '\n'

HAM = 'ham'
SPAM = 'spam'

SOURCES = [
    ('datas/beck-s', HAM),
    ('datas/farmer-d', HAM),
    ('datas/kaminski-v', HAM),
    ('datas/kitchen-l', HAM),
    ('datas/lokay-m', HAM),
    ('datas/williams-w3', HAM),
    ('datas/BG', SPAM),
    ('datas/GP', SPAM),
    ('datas/SH', SPAM)
]

SKIP_FILES = {'cmds'}


def read_files(path):
    '''
    Generator of pairs (filename, filecontent)
    for all files below path whose name is not in SKIP_FILES.
    The content of the file is of the form:
        header....
        <emptyline>
        body...
    This skips the headers and returns body only.
    '''
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


def build_data_frame(l, path, classification):
    rows = []
    index = []
    for i, (file_name, text) in enumerate(read_files(path)):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = pd.DataFrame(rows, index=index)
    return data_frame, len(rows)


def load_data():
    data = pd.DataFrame({'text': [], 'class': []})
    l = 0
    for path, classification in SOURCES:
        data_frame, nrows = build_data_frame(l, path, classification)
        data = data.append(data_frame)
        l += nrows
    data = data.reindex(np.random.permutation(data.index))
    return data

# We will load the Email spam dataset into Panadas dataframe here .
data=load_data()


data = data.sample(frac=1).reset_index(drop=True)
data = data[:10000]
print(len(data))
print(data)


# We change the dataframe index from filenames to indices here.
new_index=[x for x in range(len(data))]
data.index=new_index


print(data)

type(data)


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english",
                             encoding='latin-1',min_df=8)

train_matrix = vectorizer.fit_transform(data['text'])

print(train_matrix.shape)


import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

# def cleaner(df, col_name):
#     stop = stopwords.words('english')

#     def remove_stops(x, stopwordseq):
#         return ' '.join(x for x in x.split() if x not in stopwordseq)

#     data[col_name] = data[col_name].str.lower()\
#                                .str.replace('[^\w\s]', '').apply(remove_stops, stop)
#     return df

# data = data.pipe(cleaner, 'text')

print(data.isna())

print(data.isna().sum())
print(data.isna().sum())

duplicate_rows_df = data[data.duplicated(['text'])]
print("number of duplicate rows: ", duplicate_rows_df.shape)
print("Unique entries length: ", len(data.text.unique()))
print("Total entries length: ", len(data.text))

print("Shape of dataframe before dropping duplicates: ", data.shape)
print("shape of dataframe after dropping duplicates: ", data.drop_duplicates(subset="text").shape)


print("data types: \n", data.dtypes)


print(data)


from nltk.corpus import stopwords
stop = stopwords.words('english')

print(stop)

data['text'].apply(lambda x: [item for item in x if item not in stop])
print(data)

data['text'] = data['text'].astype(str)

print("data types: \n", data.dtypes)

#We will add two more columns to our dataframe for tokenized text and token count.
def token_count(row):
    'returns token count'
    text=row['tokenized_text']
    length=len(text.split())
    return length

def tokenize(row):
    "tokenize the text using default space tokenizer"
    text=row['text']
    lines=(line for line in text.split(NEWLINE) )
    tokenized=""
    for sentence in lines:
        tokenized+= " ".join(tok for tok in sentence.split())
    return tokenized

#We will use apply functions on dataframe to add the columns for :

#* Tokenized text
#* Token Count
#* Language
#Language column in this case is not necessary as we only have english text. However this approach is good for properly dealing with multi lingual data.



data['tokenized_text']=data.apply(tokenize, axis=1)

data['token_count']=data.apply(token_count, axis=1)


# Checking the data
data.head()

print("data types: \n", data.dtypes)

data['tokenized_text'] = data['tokenized_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

def tokenized_count(row):
    'returns token count'
    text=row['tokenized_text']
    length=len(text.split())
    return length

data['final_token_count']=data.apply(token_count, axis=1)
print(data.head())


vect = TfidfVectorizer(stop_words='english', max_df=0.50, min_df=2)
X = vect.fit_transform(data.tokenized_text)

print(X.shape)

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
X_dense = X.todense()
coords = PCA(n_components=2).fit_transform(X_dense)
plt.scatter(coords[:, 0], coords[:, 1], c='#BEF4FB')
plt.show()

coords = TruncatedSVD(n_components=2, n_iter=10).fit_transform(X)
plt.scatter(coords[:, 0], coords[:, 1], c='#000000')
plt.show()

#ignore this one
from sklearn.feature_extraction.text import TfidfVectorizer
features = vect.get_feature_names()

# print(features)
