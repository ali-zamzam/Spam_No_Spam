from sklearn import preprocessing
import numpy as np
import pandas as pd
import os
# Load required libraries
import matplotlib.pyplot as plt
import time
import pickle
import nltk
import seaborn as sns
import sys
sys.setrecursionlimit(1500)
# %matplotlib inline


# nltk.download()
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
                    past_header, lines = True, []
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
print(len(data))
print(data)


f = lambda x: ' '.join([item for item in x.split() if item not in ["Subject","Subject:","subject"]])
data["text"] = data["text"].apply(f)


# We change the dataframe index from filenames to indices here.
new_index=[x for x in range(len(data))]
data.index=new_index

print(data)


# We will load the Email spam dataset into Panadas dataframe here .

SOURCES = [
    ('datas/beck-s', HAM),
    ('datas/farmer-d', HAM),
    ('datas/kaminski-v', HAM),
    ('datas/kitchen-l', HAM),
    ('datas/lokay-m', HAM),
    ('datas/williams-w3', HAM),
]

test_data=load_data()



from nltk.corpus import stopwords
stop = stopwords.words('english')

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
#Language column in this case is not necessary as we only have english text. However this approach
# is good for properly dealing with multi lingual data.



data['tokenized_text']=data.apply(tokenize, axis=1)
data['token_count']=data.apply(token_count, axis=1)
# Checking the data
data.head()

data['tokenized_text'] = data['tokenized_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

from nltk.stem import *

from nltk.stem.porter import *
stemmer = PorterStemmer()

data['tokenized_text'] = data['tokenized_text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))


print(data.head())

import nltk
# nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


data['tokenized_text'] = data['tokenized_text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
# test_data['tokenized_text'] = test_data['tokenized_text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

print(data.head())



# Lets look at some information related to the data
df=data
print("total emails : ", len(df))
print  ("total spam emails : ", len(df[df['class']=='spam']) )
print  ("total normal emails : ", len(df[df['class']=='ham']) )


data['lang']='en'
df1 = df.groupby(['lang','class'])['class','lang'].size().unstack()

ax=df1.plot(kind='bar')
ax.set_ylabel("Total Emails")
ax.set_xlabel("Language")
ax.set_title("Plot of Emails count with languages and email type")

# We randomize the rows to subset the dataframe
df.reset_index(inplace=True)
df=df.reindex(np.random.permutation(df.index))

print(df.head())

len_unseen = 1000
df_unseen_test= df.iloc[:len_unseen]
df_model = df.iloc[len_unseen:]

print('total emails for unseen test data : ', len(df_unseen_test))
print('\t total spam emails for enron  : ', len(df_unseen_test[(df_unseen_test['lang']=='en') & (df_unseen_test['class']=='spam')]))
print('\t total normal emails for enron  : ', len(df_unseen_test[(df_unseen_test['lang']=='en') & (df_unseen_test['class']=='ham')]))
print()

print('total emails for model training/validation : ', len(df_model))
print('\t total spam emails for enron  : ', len(df_model[(df_model['lang']=='en') & (df_model['class']=='spam')]))
print('\t total normal emails for enron  : ', len(df_model[(df_model['lang']=='en') & (df_model['class']=='ham')]))


# Load required libraries
import matplotlib.pyplot as plt
bins = [0,100,200,300,350,400,500,600,800,1000,1500,2000,3000,4000,5000,6000,10000,20000]


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))
fig.subplots_adjust(hspace=.5)

df_sub=df[ (df['lang']=='en') & (df['class']=='ham')]
df1 = df_sub.groupby(pd.cut(df_sub['token_count'], bins=bins)).token_count.count()
df1.index=[a.right for a in df1.index]
res1=df1.plot(kind='bar',ax=axes[0])
res1.set_xlabel('Email tokens length')
res1.set_ylabel('Frequency')
res1.set_title('Token length Vs Frequency for Enron Normal Emails')


df_sub=df[ (df['lang']=='en') & (df['class']=='spam')]
df1 = df_sub.groupby(pd.cut(df_sub['token_count'], bins=bins)).token_count.count()
df1.index=[a.right for a in df1.index]
res2=df1.plot(kind='bar',ax=axes[1])
res2.set_xlabel('Email tokens length')
res2.set_ylabel('Frequency')
res2.set_title('Token length Vs Frequency for Enron Spam Emails')

import keras

from keras.layers import Input, Dense
from keras.models import Model,load_model
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard

import joblib
import sklearn
from sklearn import metrics
from sklearn import svm
# from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder


# max number of features
num_max = 4000

import time


def train_tf_idf_model(texts):
    "train tf idf model "
    tic = time.process_time()

    tok = Tokenizer(num_words=num_max)
    tok.fit_on_texts(texts)
    toc = time.process_time()

    print(" -----total Computation time = " + str((toc - tic)) + " seconds")
    return tok


def prepare_model_input(tfidf_model, dataframe, mode='tfidf'):
    "function to prepare data input features using tfidf model"
    tic = time.process_time()
    le = LabelEncoder()
    sample_texts = list(dataframe['tokenized_text'])
    sample_texts = [' '.join(x.split()) for x in sample_texts]

    targets = list(dataframe['class'])
    targets = [1. if x == 'spam' else 0. for x in targets]
    sample_target = le.fit_transform(targets)

    if mode == 'tfidf':
        sample_texts = tfidf_model.texts_to_matrix(sample_texts, mode='tfidf')
    else:
        sample_texts = tfidf_model.texts_to_matrix(sample_texts)

    toc = time.process_time()

    print('shape of Class: ', sample_target.shape)
    print('shape of data: ', sample_texts.shape)

    print(" -----total Computation time for preparing model data = " + str((toc - tic)) + " seconds")

    return sample_texts, sample_target

texts=list(df_model['tokenized_text'])
tfidf_model=train_tf_idf_model(texts)

# prepare model input data
mat_texts,tags=prepare_model_input(tfidf_model,df_model,mode='tfidf')

#Splitting Training and Testing Data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(mat_texts, tags, test_size=0.40)
print ('train data shape: ', X_train.shape, y_train.shape)
print ('validation data shape :' , X_val.shape, y_val.shape)

## Define and initialize the network

model_save_path="modele/spam_detector_enron_model.h5"


import datetime
def get_simple_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(num_max,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc',keras.metrics.binary_accuracy])
    print('compile done')
    return model

def check_model(model,x,y,epochs=2):
    history=model.fit(x,y,batch_size=32,epochs=epochs,verbose=1,shuffle=True,validation_split=0.2,
              callbacks=[checkpointer, tensorboard]).history
    return history


def check_model2(model,x_train,y_train,x_val,y_val,epochs=10):
    history=model.fit(x_train,y_train,batch_size=64,
                      epochs=epochs,verbose=1,
                      shuffle=True,
                      validation_data=(x_val, y_val)).history
    return history

# define checkpointer
checkpointer = ModelCheckpoint(filepath=model_save_path,
                               verbose=1,
                               save_best_only=True)


# define tensorboard
tensorboard = TensorBoard(log_dir = os.path.join("logs","fit",datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)




# define the predict function for the deep learning model for later use
def predict(data):
    result=model.predict(data)
    prediction = [round(x[0]) for x in result]
    return prediction

# get the compiled model
model = get_simple_model()

# load history
# history=check_model(m,mat_texts,tags,epochs=10)
history=check_model2(model,X_train,y_train,X_val,y_val,epochs=10)

# plot the loss on train and validation data
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Email Spam Filter Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');


##RANDOM FOREST MODEL
from sklearn.ensemble import RandomForestClassifier
spam_model_rf = RandomForestClassifier(n_estimators=1000,random_state=5)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
spam_model_rf.fit(X_train,y_train)

##EVALUATE MODEL PERFORMANCE
sample_texts,sample_target=prepare_model_input(tfidf_model,df_unseen_test,mode='')


model_dict={}
model_dict['random_forest']=spam_model_rf
model_dict['neural_networks']=model


def getResults(model_dict, sample_texts, sample_target):
    '''
    Get results from different models
    '''
    results = []

    results_cm = {}

    for name, model in model_dict.items():
        #         print(name)
        tic1 = time.process_time()
        if name in 'neural_networks':
            predicted_sample = predict(sample_texts)
            cm = sklearn.metrics.confusion_matrix(sample_target, predicted_sample)
        else:
            predicted_sample = model.predict(sample_texts)
            cm = sklearn.metrics.confusion_matrix(sample_target, predicted_sample.round())

        toc1 = time.process_time()
        #         print(predicted_sample)

        results_cm[name] = cm

        total = len(predicted_sample)
        TP = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[1][1]

        time_taken = round(toc1 - tic1, 4)
        res = sklearn.metrics.precision_recall_fscore_support(sample_target, predicted_sample)
        accuracy = sklearn.metrics.accuracy_score(sample_target, predicted_sample)
        results.append(
            [name, np.mean(res[0]), np.mean(res[1]), np.mean(res[2]), accuracy, TP, FP, FN, TN, str(time_taken)])

    df_cols = ['model', 'precision', 'recall', 'f1_score', 'Accuracy', 'TP', 'FP', 'FN', 'TN', 'execution_time']
    result_df = pd.DataFrame(results, columns=df_cols)

    return result_df, results_cm

result_df,results_cm= getResults(model_dict,sample_texts,sample_target)
print(result_df)

#NN SMR AND HMR COMPUTATION
result_df.iloc[0][5]
HMR = float(result_df.iloc[0][6] / (result_df.iloc[0][8] + result_df.iloc[0][6]))
SMR = float(result_df.iloc[0][7] / (result_df.iloc[0][5] + result_df.iloc[0][7]))

print("RANDOM FOREST:")
print("HAM Misclassification Rate: ",HMR)
print("SPAM Misclassification Rate: ", SMR)

#NN SMR AND HMR COMPUTATION
result_df.iloc[0][5]
HMR = float(result_df.iloc[1][6] / (result_df.iloc[1][8] + result_df.iloc[1][6]))
SMR = float(result_df.iloc[1][7] / (result_df.iloc[1][5] + result_df.iloc[1][7]))

print("NEURAL NETWORK:")
print("HAM Misclassification Rate: ",HMR)
print("SPAM Misclassification Rate: ", SMR)


def plot_heatmap(cm, title):
    df_cm2 = pd.DataFrame(cm, index=['normal', 'spam'])
    df_cm2.columns = ['normal', 'spam']

    ax = plt.axes()
    sns.heatmap(df_cm2, annot=True, fmt="d", linewidths=.5, ax=ax)
    ax.set_title(title)
    plt.show()

    return

plot_heatmap(results_cm['neural_networks'],'Neural Networks')

plot_heatmap(results_cm['random_forest'],'Random Forest')
