# Machine Learning Engineer Nanodegree
## Capstone Project
Sujay Bhowmick  
August 9th, 2018

## I. Definition
### Project Overview
In NLP Sentiment Analysis is a process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative or neutral.

Sentiment Analysis is useful in many ways. In my use case we are classifying Tweets of some relevance to domain like Finance related to both Micro and Macro events for which various credible financial analyst, activist, famous investors and financial news publishers are talking about in Twitter through their twitter handles (can not provide the Twitter handles of the users here due to privacy concerns). These Tweets are then used to guage the sentiment of the investors on certain topics and can be used to assess the investment decisions on that particular financial asset or stock after sentiments are combined with other analytical data dervided using different methodologies.

I am using a Machine Learning based approach to solve this problem and will develop a classification model, which is trained using prelabeled dataset of **positive**, **negative** content of the Twitter Tweets

### Problem Statement
The goal of this Machine Learning Nanodegree Capstone project is to analyse the sentiment of various twitter tweets which is publicly available.

The tweets are related to financial news which have been labelled by a human for training and testing purpose of the Machine Learning model. There are approximately 8000 tweets which have been labelled with labels positive and negative for tweets indicating a **positive** sentiment and **negative** sentiment respectively.

### Metrics
I have chosen accuracy as my metrics to assess whether model is working as expected. My training dataset is unbalanced between **positive** and **negative** labels (with *3843 positive* and *4508 negative* tweets), hence I would also like to look at confusion matrix and determine if whether the model is working as expected by looking at additional metrics like precision, recall and F1-Score


## II. Analysis
### Data Exploration

**Data Collection**

I have collected financially relevant messages from Twitter (had to do parallel project to make sure I only try to get financially relevant tweets, but its outside the scope of this project as it was used only to collected the data needed to train the model).

**Data Preprocessing**

Most of the tweets contains twitter handles (e.g. @sujay), hash tags, hyperlinks. Hence I need to preprocess the data  and and replace each of them with normalized tags suchs as <LINK/> for hyperlinks, <HASHTAG/> for hash tags, twitter handle with <NAME/>, special entities like quotes, ampersand and other various special characters which is outside the characted set of English

After preprocessing following are the dataset break of messages after labelling them with sentiment label

```html
Total Labelled Messages: 8351
Positive Labels: 3843
Negative Labels: 4508
```

Link for the labelled dataset can be found [here](https://github.com/sujaybhowmick/twitter_sentiment_analysis/blob/master/data/preprocessed_tweets.csv)

### Exploratory Visualization

Following data is loaded using Pandas and explored. Below is Pandas Dataframe of the dataset

|      | content                                          | label |
| ---- | ------------------------------------------------ | ----- |
| 0    | boeing hit hard by tariff and trade war headl... | 0     |
| 1    | <NAME/> <HASHTAG/> microsoft is a proud spons... | 1     |
| 2    | <NAME/> <NAME/> it 's not fake news , i own b... | 1     |
| 3    | <NAME/> canada should consider slapping 300% ... | 0     |
| 4    | 'upwards of 20 , 00 workers' could lose jobs ... | 0     |
| 5    | $tsla short interest: 28 , 382 , 800 vs prev ... | 1     |
| 6    | the most logical way-forward for <HASHTAG/> s... | 0     |
| 7    | <NAME/> <NAME/> or could lead to a monopoly w... | 1     |
| 8    | we need to break up google , disney , and eve... | 0     |
| 9    | venkatesh potluri , a research fellow at micr... | 1     |
| 10   | venkatesh potluri , a research fellow at micr... | 1     |
| 11   | veolia teams mobilized to restore <HASHTAG/> ... | 1     |
| 12   | favoring insider deal , beach leaders \( town... | 0     |
| 13   | insider trade update: jarl berntzen increases... | 1     |
| 14   | investigate salesforce insider trading crime ... | 0     |
| 15   | <NAME/> <NAME/> <HASHTAG/> podcast: avaya to ... | 1     |
| 16   | <HASHTAG/> breakingnews <HASHTAG/> tech magic... | 1     |
| 17   | rt <NAME/> ironic that freeport mcmoran is th... | 1     |
| 18   | rt <NAME/> exciting to hear that my companyêl... | 1     |
| 19   | not great reading for waitrose , heading in t... | 0     |
| 20   | fitbit 's looking for a sweet turnaround - th... | 1     |
| 21   | kimberly-clark wins 2018 climate leadership a... | 1     |
| 22   | per wsj , unilever threatens removing adverti... | 0     |
| 23   | unilever calls out facebook/google sexism , r... | 0     |
| 24   | coffee 's on: 41st street starbucks reopens a... | 1     |
| 25   | experts: "starbucks ceo schultz 's hiring of ... | 0     |
| 26   | datacentrix takes top honours at 2017 hpe par... | 1     |
| 27   | mcdonald 's , hedging their bets , under-orde... | 0     |
| 28   | rt <NAME/> arby 's buys buffalo wild wings , ... | 0     |
| 29   | rt <NAME/> due to the forecasted heavy snow ,... | 0     |
| ...  | ...                                              | ...   |
| 8321 | abbvie 's hepatitis c drug , novartis' lung c... | 1     |
| 8322 | abbvie 's hepatitis c drug , novartis' lung c... | 1     |
| 8323 | vw must recall around 57 , 600 of its diesel ... | 0     |
| 8324 | multiple <HASHTAG/> myeloma study results enc... | 1     |
| 8325 | "wpp teamed up with cambridge analytica to wo... | 1     |
| 8326 | ingram micro expands cybersecurity capabiliti... | 1     |
| 8327 | branded a liar \?  the <HASHTAG/> volkswagen...  | 0     |
| 8328 | our collaboration with the johnson & johnson ... | 1     |
| 8329 | looking to become a <HASHTAG/> pinterest rock... | 1     |
| 8330 | the immoral minority: pepsico reserves one hu... | 1     |
| 8331 | <NAME/> source says the layoff wo n't include... | 0     |
| 8332 | rt <NAME/> breaking: british members of parli... | 0     |
| 8333 | carb:carbonitepaying145m to buy dell sub...      | 1     |
| 8334 | rt <NAME/> netflix acquires sundance award-wi... | 1     |
| 8335 | <NAME/> <HASHTAG/> trump gave special permits... | 0     |
| 8336 | rt <NAME/> <HASHTAG/> aadhaar identity fraud ... | 0     |
| 8337 | amznpartneringwjpm , berkshire hathaway ...      | 1     |
| 8338 | amznpartneringwjpm , berkshire hathaway ...      | 1     |
| 8339 | now this is truly disruptive. the spinoffs of... | 0     |
| 8340 | now this is truly disruptive. the spinoffs of... | 0     |
| 8341 | <NAME/> has warned customers to be cautious o... | 0     |
| 8342 | stop illegal mass layoff in verizon: <HASHTAG... | 0     |
| 8343 | i will keep everyone posted as i find out mor... | 0     |
| 8344 | rt <NAME/> we are excited <NAME/> to <HASHTAG... | 1     |
| 8345 | <HASHTAG/> rnaseq veracyte plans two new test... | 1     |
| 8346 | rt <NAME/> how apple pay can make credit card... | 0     |
| 8347 | globe , disney partner to support hero founda... | 1     |
| 8348 | rt <NAME/> fantastic story of partnership wit... | 1     |
| 8349 | messaging on net neutrality is all messed up.... | 0     |
| 8350 | messaging on net neutrality is all messed up.... | 0     |

8351 rows × 2 columns

**Feature extraction**

The below graph represents the distribution of token per sentence in the dataset samples. A custom tokenizer using Keras text preprocessing tokenizer is used to observe the distribution of words. We can then determine the maximum number of tokens in the training dataset. This is a good input feature which can be used for building the classifier. 

![img](data:image/png;base64,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 )

### Algorithms and Techniques

I have decided to use 1-D Convolution Neural to train the model on the training dataset. CNN model is very popular in image classification and very recently has show lot of success in text classification and natural language processing. 

One of the desirable properties of CNN is that it preserves 2D spatial orientation in computer vision. Texts, like pictures, have an orientation. Instead of 2-dimensional, texts have a one-dimensional structure where words sequence matter. Words in the sentence are each replaced by a n-dimensional word vector, hence we fix one dimension of the filter to match the word vectors and vary the region size, h. Region size refers to the number of rows – representing word – of the sentence matrix that would be filtered. This is basic idea on how CNN can be useful for NLP and text classification. 

For training and validation I am using train_test_split on the data set using sklearn to create the training and testing data sets.

### Benchmark
The benchmark for this model is Afinn model which is currently in use but suffers from low accuracy of around 51%. 

<u>How Afinn model works</u>

1. Methodology is keyword-matching,
2. Dictionaries of keywords and their Sentiment Value are pre-defined,
3. Input message is split by all non-alphanumeric characters into individual Tokens,
4. Each token is matched against the dictionary in the appropriate language,
5. Afinn's Sentiment Score is the sum of all Sentiment Values of the matched Tokens in the input message.

<u>Afinn's weakness</u>

1. Dictionary is more suited for analyzing product reviews
2. Methodology cannot reliably deal with even slightly complex language patterns (e.g. "not good")

The score from Afinn model is an integer value formula as Afinn returns the sum of values of all tokens in a message. Based on a small some test, this type of score made it more difficult to translate from Sentiment Score to Sentiment Label as the results may vary a lot.


## III. Methodology
### Data Preprocessing
Data processing is part of the data analysis, the details of which is provided in the Data Analysis -> Data Exploration section, kindly refer to the same

### Implementation

**CNN Model**

I have decided to use Convolutional Neural Network (CNN) classifier to predict the sentiment (positive or negative) of a tweet

Following is the architecture diagram of the 1-D CNN which is implemented in this project and used

![architecture](/Users/sujaybhowmick/development/courses/mlnd/MLND-Capstone/twitter-sentiment-analysis/docs/img/CNN_Architecture.png)

Kim Yoon’s [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) as a reference architecture

Below is the model summary and plot

```html
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_19 (Embedding)     (None, 22, 40)            1200000   
_________________________________________________________________
dropout_37 (Dropout)         (None, 22, 40)            0         
_________________________________________________________________
conv1d_19 (Conv1D)           (None, 20, 250)           30250     
_________________________________________________________________
global_max_pooling1d_19 (Glo (None, 250)               0         
_________________________________________________________________
dense_37 (Dense)             (None, 250)               62750     
_________________________________________________________________
dropout_38 (Dropout)         (None, 250)               0         
_________________________________________________________________
activation_37 (Activation)   (None, 250)               0         
_________________________________________________________________
dense_38 (Dense)             (None, 1)                 251       
_________________________________________________________________
activation_38 (Activation)   (None, 1)                 0         
=================================================================
Total params: 1,293,251
Trainable params: 1,293,251
Non-trainable params: 0
_________________________________________________________________
```

![model](/Users/sujaybhowmick/development/courses/mlnd/MLND-Capstone/twitter-sentiment-analysis/docs/img/model_plot.png)

```python
def get_model():

    # CNN Model

    NUM_FILTERS = 250
    KERNEL_SIZE = 3
    HIDDEN_DIMS = 250

    model = Sequential()

    # We use embedding layer which maps our vocabulary indices into EMBEDDING_DIM 	    dimensions
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(Dropout(0.2))

    # Adding Convolution1D
    model.add(Conv1D(NUM_FILTERS,
                     KERNEL_SIZE,
                     padding='valid',
                     activation='relu',
                     strides=1))

    # Add a max pooling:
    model.add(GlobalMaxPooling1D())

    # Add a simple hidden layer:
    model.add(Dense(HIDDEN_DIMS))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and use sigmoid function
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

**Word Tokenizer**

Before using the data for the model, we need to process the tweet content to equivalent word vector. For this purpose we will use Keras Tokenizer to convert each word into a corresponding integer identifier. In order for us to use the content in the Model we must ensure the length of the content is same. We can do this by using the Keras **sequence.pad_sequences** function. All content greater than MAX_LEN will be truncated and text which are less than MAX_LEN will be padded to get the same length.

```python
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Flatten, Conv1D, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from keras.utils.vis_utils import plot_model
from tensorflow import set_random_seed
from numpy.random import seed
seed(1)
set_random_seed(2)

tweet_tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tweet_tokenizer.fit_on_texts(df['content'].values)
X = tweet_tokenizer.texts_to_sequences(df['content'].values)

X = sequence.pad_sequences(X, maxlen=MAX_LEN, padding="post", value=0)

y = df['label']

def print_sample_before_after_tokenizing():
    print('First sample before preprocessing: \n', df['content'].values[0], '\n')
    print('First sample after preprocessing: \n', X[0])
    
print_sample_before_after_tokenizing()
```

### Refinement
Since I have limited labelled data, the earlier model was running on dataset which is split 80:20 between training and test set and generally suffered from high bias. The accuracy of the model was about 79%. In order to overcome the limitation of the model, I decided to use corss validation technique to reduce bias and also reduces variance as most of the data is also being used in validation set.

Also adjusting the parameter MAX_LENGTH (max length of the tokens) in the data set helps in increasing accuracy of the model as observed during pre refinement activity. 


## IV. Results
### Model Evaluation and Validation

We applied CNN to the final model and has shown improved results and reported a validation accuracy score of 79%. Also changing some of the hyper parameters did not affect the accuracy score and confusion matrix of the model.

The Hyperparameter values which holds good for the current model.

```python
# Number of examples used in each iteration
BATCH_SIZE = 32 
# Size of vocabulary dictionary
VOCAB_SIZE = 30000
# Max length of tweet as per the plot above
MAX_LEN = 22
# Dimension of word embedding vector
EMBEDDING_DIM = 40
```

```python
SENTIMENT_LABELS = ['negative', 'positive']

filepath="models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

    
EPOCHS = 2 # Number of passes through entire dataset
history = model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.1, callbacks=callback_list, verbose=0)

# Evaluate the model
score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print('\nAccuracy: ', acc*100)

pred_classes = model.predict_classes(X_test)

plot_confusion_matrix(y_test, pred_classes)
```

The model is tested with various inputs and has an accuracy of 79% . Further we have the confusion matrix which gives us better idea of our classification model 

```python
Epoch 00001: val_acc improved from -inf to 0.74102, saving model to models/weights-improvement-01-0.74.hdf5

Epoch 00002: val_acc improved from 0.74102 to 0.79940, saving model to models/weights-improvement-02-0.80.hdf5
1671/1671 [==============================] - 0s 44us/step
Accuracy:  79.83243566726512
```

**Confusion Matrix**:



![img](data:image/png;base64,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 )

```python
 precision    recall  f1-score   support

   negative       0.81      0.80      0.80       905
   positive       0.76      0.78      0.77       766

avg / total       0.79      0.79      0.79      1671
```



## V. Conclusion

### Free-Form Visualization

Below code predicts Sentiment Lables and compares it to Human provided lables and based on prediction results of few tweets and seems like my model is doing much better than the benchmark model

```python
SENTIMENT_LABELS = ['negative', 'positive']
def get_prediction(tweet):
    # Preprocessing step
    tweet_words_array = tweet_tokenizer.texts_to_sequences([tweet])
    tweet_words_array = sequence.pad_sequences(tweet_words_array, maxlen=MAX_LEN, padding="post", value=0)
    
    #Predict the sentiment label and score
    score = model.predict(tweet_words_array)[0][0]
    prediction = SENTIMENT_LABELS[model.predict_classes(tweet_words_array)[0][0]]
    print('Tweet:', tweet, '\nPrediction:', prediction, '\nScore: ', score)
    print('\n')
    return prediction, score

# Test Prediction
prediction = get_prediction(". RT @SpryGuy: The CEO of Papa John's stiffs and cheats his own employees so he can live in this castle with a moat. NEVER buy Papa John's pi…")
assert prediction[0] == "negative"
prediction = get_prediction(". GVC Holdings consummated the acquisition of Ladbrokes Coral https://t.co/xaN4ACA0h6 https://t.co/ZNm0gmXLK7")
assert prediction[0] == "positive"
prediction = get_prediction(". family fully prepared to drop Roku, Apple iPhones , Amazon Prime, toss out Alexa,Google emails chromes, etc in the… https://t.co/64cZYuhYSQ")
assert prediction[0] == "negative"
prediction = get_prediction(". #AtlasMara holding is a real ingenious feat in The financial fraternity..am amazed at the forge ahead they posses.. #mindblown")
assert prediction[0] == "positive"
prediction = get_prediction(". Boeing hit hard by tariff and trade war headlines today, down -3.5%. Also note the very ugly price/momentum diverge https://t.co/h9bfT95yWZ")
assert prediction[0] == "negative"
prediction = get_prediction("RT @CentroneInvests: \"Be prepared 4 the crash of the dollar invest in precious metals for security!\" #Invest4Success #Investors ??on eBay ht…")
assert prediction[0] == "negative"
prediction = get_prediction("Didn't see this one coming but makes so much sense... Amazon to Buy Whole Foods in $13.4 Billion Deal https://t.co/tKcF9dUwct")
assert prediction[0] == "positive"
prediction = get_prediction("Starbucks Corporation (SBUX) Stock Isn't as Bad as it Looks. Starbucks Corporation (Nasdaq: SBUX) is making aggressive changes to get its stock back on track. The latest change the company announced this week is the departure of CFO Scott Maw, and analysts say")
assert prediction[0] == "positive"
```

```python
Tweet: . RT @SpryGuy: The CEO of Papa John's stiffs and cheats his own employees so he can live in this castle with a moat. NEVER buy Papa John's pi… 
Prediction: negative 
Score:  0.07339549


Tweet: . GVC Holdings consummated the acquisition of Ladbrokes Coral https://t.co/xaN4ACA0h6 https://t.co/ZNm0gmXLK7 
Prediction: positive 
Score:  0.9204503


Tweet: . family fully prepared to drop Roku, Apple iPhones , Amazon Prime, toss out Alexa,Google emails chromes, etc in the… https://t.co/64cZYuhYSQ 
Prediction: negative 
Score:  0.068363935


Tweet: . #AtlasMara holding is a real ingenious feat in The financial fraternity..am amazed at the forge ahead they posses.. #mindblown 
Prediction: positive 
Score:  0.66840297


Tweet: . Boeing hit hard by tariff and trade war headlines today, down -3.5%. Also note the very ugly price/momentum diverge https://t.co/h9bfT95yWZ 
Prediction: negative 
Score:  0.19279152


Tweet: RT @CentroneInvests: "Be prepared 4 the crash of the dollar invest in precious metals for security!" #Invest4Success #Investors ??on eBay ht… 
Prediction: negative 
Score:  0.039829064


Tweet: Didn't see this one coming but makes so much sense... Amazon to Buy Whole Foods in $13.4 Billion Deal https://t.co/tKcF9dUwct 
Prediction: positive 
Score:  0.87018085


Tweet: Starbucks Corporation (SBUX) Stock Isn't as Bad as it Looks. Starbucks Corporation (Nasdaq: SBUX) is making aggressive changes to get its stock back on track. The latest change the company announced this week is the departure of CFO Scott Maw, and analysts say 
Prediction: positive 
Score:  0.7714823
```

As you can see our model predicted all the labels correctly for few sample tests of the data. 

### Reflection

The most important thing in this project I understood about the concept of overfitting or undercutting in some case. 

Initially I decied to use KFolds for training and testing my model, but quickly the model started to show signs of overfitting, but the test validation was stable. I realised that my model will not generalize well.

I then decided to try another method for splitting the data and use the training set and test set to measure the  metrics of the preformance and my model is not overfitting anymore and 

### Improvement

With more labeled data and with help of new architecture such RNN, we can definitely improve the quality of  our model.  I have not used RNN, but I was suggested to try using RNN for Sentiment Analysis and not CNN which is essential is a model know to generally used for image classification.

-----------

