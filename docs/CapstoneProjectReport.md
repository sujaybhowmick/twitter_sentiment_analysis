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

### Exploratory Visualization

Following data is loaded using Pandas and explored. Below is Pandas Dataframe of the dataset

| content | label                                             |      |
| ------- | ------------------------------------------------- | ---- |
| 0       | omg its already 7:30 :O                           | 1    |
| 1       | Juuuuuuuuuuuuuuuuussssst Chillin!!                | 1    |
| 2       | handed in my uniform today . i miss you already   | 1    |
| 3       | hmmmm.... i wonder how she my number <NAME/>      | 1    |
| 4       | thanks to all the haters up in my face all day... | 1    |
| 5       | Feeling strangely fine. Now I'm gonna go liste... | 1    |
| 6       | You're the only one who can see this cause no ... | 1    |
| 7       | goodbye exams, HELLO ALCOHOL TONIGHT              | 1    |
| 8       | uploading pictures on friendster                  | 1    |
| 9       | (: !!!!!! - so i wrote something last week. an... | 1    |
| 10      | ... Do I need to even say it? Do I? Well, her...  | 1    |
| 11      | ... health class (what a joke!)                   | 1    |
| 12      | <NAME/> < 3 GO TO THE SHOW TONIGHT                | 1    |
| 13      | bathroom is clean..... now on to more enjoyabl... | 1    |
| 14      | boom boom pow                                     | 1    |
| 15      | go give ur mom a hug right now. <LINK/>           | 1    |
| 16      | Going To See Harry Sunday Happiness               | 1    |
| 17      | - I always get what I want                        | 1    |
| 18      | I bend backwards                                  | 1    |
| 19      | i get off work sooooon! i miss cody booo. have... | 1    |
| 20      | I hate allergies. Should I get my hair cut tom... | 1    |
| 21      | I'm really going to bed now...                    | 1    |
| 22      | Jin has a twitter.                                | 1    |
| 23      | just gonna smile...cuz it is what it is..and i... | 1    |
| 24      | Just got home, and I got to see my friend Zah...  | 1    |
| 25      | oh thank you!                                     | 1    |
| 26      | pleased                                           | 1    |
| 27      | Rose and ood will be back in the Xmas Who spec... | 1    |
| 28      | Thanks, I need all the help i can get.            | 1    |
| 29      | - that explains alot.                             | 1    |
| ...     | ...                                               | ...  |
| 1552970 | Yummy! Bummer...on my way to mom's for a famil... | 0    |
| 1552971 | Yummy, jess, your doingg everythingg I wanna b... | 0    |
| 1552972 | Yup - late for work! But not nearly as late as... | 0    |
| 1552973 | yup all alone on my bday happy bday every1 bor... | 0    |
| 1552974 | yup four more days to they get back and six mo... | 0    |
| 1552975 | yup gots a headache                               | 0    |
| 1552976 | yup the crying has started. Finished part I an... | 0    |
| 1552977 | Yup! So, they did take my phone. But, it will ... | 0    |
| 1552978 | Yup, I was right. I'm a sick girl, damn           | 0    |
| 1552979 | Yup, Jr's collarbone is broken I can tell he i... | 0    |
| 1552980 | yup, my cat is still stranded she's just sitti... | 0    |
| 1552981 | Yup, needed the tissues for BSG but not quite ... | 0    |
| 1552982 | Yup, SPORE is updating now                        | 0    |
| 1552983 | Yup, that is all                                  | 0    |
| 1552984 | Yup, the hints yesterday were correct. I'm now... | 0    |
| 1552985 | Yup. Bawling like a baby. So sad                  | 0    |
| 1552986 | Yup. Total sausage fest at the bar tonight. Ol... | 0    |
| 1552987 | Yupp. I'm going to miss have to frees a day       | 0    |
| 1552988 | yuppies are real weird when you say " excuse ...  | 0    |
| 1552989 | Yvonne left House is going to be in shambles n... | 0    |
| 1552990 | yzabellopez: ÃÂ yayayay cant wait, i just wis...  | 0    |
| 1552991 | ze mother has spoken.. no playoffs fo' kathy t... | 0    |
| 1552992 | Zefron and BBV, please dont break-up.             | 0    |
| 1552993 | Zese v posteli. Posloucham System of a down a ... | 0    |
| 1552994 | Zeta is getting old and I dont want her to go ... | 0    |
| 1552995 | Zicam Cold Remedy made my nose bleed i don't r... | 0    |
| 1552996 | zicam is being pulled from market!!! oh, the m... | 0    |
| 1552997 | Zigs the cat <NAME/> drat! you didn't win         | 0    |
| 1552998 | Zip Lining today in Monteverde. Only 4 days left  | 0    |
| 1552999 | Zipper flower fail Moving on to next project. ... | 0    |

1553000 rows × 2 columns

**Feature extraction**

The below graph represents the distribution of token per sentence in the dataset samples. A custom tokenizer using Keras text preprocessing tokenizer is used to observe the distribution of words. We can then determine the maximum number of tokens in the training dataset. This is a good input feature which can be used for building the classifier. 

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZsAAAEWCAYAAACwtjr+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAAIABJREFUeJzt3Xu8VXWd//HXW/BCKgGKRkCBSRfH33g7KWW/xtIU7YJdnHQsyJ/zoxwtu0xFNRON1e9h/kqNyWgoUeiiqZmiqMigdvMSxyTFC3FUlJMoR0HELAv9zB/rc3J52mefzdF1Nmzez8djPfZan/W97cWGD+uyv1sRgZmZWZW2afYAzMys9TnZmJlZ5ZxszMysck42ZmZWOScbMzOrnJONmZlVzsnGWp6kSZI6mj0Os62Zk41tESQ9WVqelfTH0vbxzR5fK5J0s6QPNHsc1hoGN3sAZo2IiJ261yWtBP45Iv67eSNqDkmDI2Jjs8dhtql8ZmMtQdIQSedIWi2pU9L/l7RtL2U/Lel2SS/L7Xfn9uOSfiFpr1LZhyV9QtIySesl/VDSdrnvZZKuyXqPSbqul/52kBSSTpG0UlKXpK9KUqnMhyUtl7RW0gJJo3vUPUnSvcCyGu3vKOnCrPu4pFskDc99IyTNy/exStIMSdvkvo9IWixpZta7V9Jhue8bwOuB7+XZ4zcyvrek6yStk3S3pKNL47hQ0tmSFkraIOlXkl5Z2r9Pqe7Dkj6V8UGS/l3SfZIezWM8rKE/eNtyRIQXL1vUAqwEDusROwP4BbArsDuwBPhC7psEdOT6V4FbgBG5PRFYDRwADAKmAb8DBuf+h4FfZZsjgQ7gQ7nvLOCbFFcItgPe3Mt4dwACWAgMA8YD9wEfyP3HAncDrwa2Bb4CXN+j7oKsO6RG+6cClwBDciyvB3bMfVcD/wm8BBgF3AZMzX0fAf4CTMn3/glgZandm7vHmNtD81gdn+VfD6wF9sz9FwJrgP3zfVwCnJ/7hgNdwCnA9tnW63Pf9Pyze3m+3/OB85r9OfPyIv+9bfYAvHjZ1KWXZPN74K2l7cnAPbk+CbgXOAe4Hti5VO687qRUij0AHJTrDwPvK+2bCZyd62cAFwN79DHe7oRxSCn2SWBBrl8PHF/at20mgd1Ldd9Yp/1/AX4G7N0j/krgD8C2pdgJwNW5/hFgWWnfiOxrWG73TDZTgUU9+pgLfDbXLwS+Vdr3HmBpqd+behn//cDBpe3xwFOAmv1Z8/LiLb5nY1u8vBz1Mook0e0BYHRpezeKf/DeGREbSvFXAv8o6dOl2HY96j5cWn+K4uwJirOk04DrJf0F+HZEnFlnqKt6jO/lpTF8R9I5pf0bgTHA+hp1ezqX4v1fImknYB7w79nuDkBX6YrdNhRnZ729N4CdgMdr9PNK4M2SyvsGA+vqtNd9r20sRcJ/nvyzGwtcJak8K/A2wC7AozXGYVsgJxvb4kVESHqY4h/D7n/QXkFxttPtEYozgB9JemdE/DrjqyjOML7Rj37XU1zCOlXSPhRJ55aI+FUvVcr/4L4CeKg0hk9HxE96VpC0Q3d3dcbxNPBF4IuS9qC4XHcncCPwJDA88pRhE/Wsswq4NiLe2Y+2VgFH/k0HxZ/d74H3RMSt/WjXthB+QMBaxQXADEm7SNoN+ALwg3KBiLgW+D/AFZL2y/Bs4KOS2lTYSdK7JL2krw6z3Pj83/l64JlcevNZSS+VNI7i3sWPM/4d4N8kvSbbHS7pvQ2+byQdJmmvvPH/BMVZ0TMRcT/FpbAzJO0saRtJEyS9qcGmHwH2KG1fBuwn6f2StpW0naSJkl7dQFuXAXvmgw7bSRoq6fW57zvA6ZLG5vvZTVJ/EpptxpxsrFV8EbiL4n/0Sylu6p/Rs1BELKC4V3G1pL/Ps5CPAf9Fcenod8A/UedMouR1wA3ABuDnwNcj4uY65RcAvwXaKe71/CDHdAHwLeBSSU/k+N/WQP/dRgOX5ziWAVcBF+W+4ygeLLiH4mb+jynuBTXiLGBKPj12RkSsA46guBy5muLM7CsU95jqyrpvo3gYYg2wHOhOemcA/w1cJ2kDxRnZ/g2O0bYQ6t/ZtZk1Ki+F/REYGxGdzR6PWTP4zMbMzCrnZGNmZpXzZTQzM6ucz2zMzKxy/p5N2nXXXWPcuHHNHoaZ2Rbl1ltvfTQiRvZVzskmjRs3jvb29mYPw8xsiyLpgb5L+TKamZkNACcbMzOrnJONmZlVzsnGzMwq52RjZmaVc7IxM7PKOdmYmVnlnGzMzKxyTjZmZlY5zyDwIhg3fUHN+MrT3z7AIzEz2zz5zMbMzCpXWbKR9BpJS0vLE5I+LmmEpEWSVuTr8CwvSTMldUi6XdL+pbamZvkVkqaW4gdIuiPrzMzfgqe3PszMrDkqSzYRsTwi9o2IfYEDgKeAnwLTgcURMQFYnNsARwITcpkGzIIicQAzgIOAA4EZpeQxK8t215uU8d76MDOzJhioy2iHAvdGxAPAZGBuxucCR+f6ZGBeFG4GhkkaBRwBLIqItRGxDlgETMp9QyPipih+AW5ej7Zq9WFmZk0wUMnmWOCCXN89IlYD5OtuGR8NrCrV6cxYvXhnjXi9Pp5H0jRJ7ZLau7q6+vnWzMysL5UnG0nbAe8CLu6raI1Y9CPesIiYHRFtEdE2cmSfv/1jZmb9NBBnNkcCv4mIR3L7kbwERr6uyXgnMLZUbwzwUB/xMTXi9fowM7MmGIhkcxzPXUIDmA90P1E2Fbi8FJ+ST6VNBNbnJbCFwOGShueDAYcDC3PfBkkT8ym0KT3aqtWHmZk1QaVf6pT0EuBtwIdL4dOBiySdCDwIHJPxq4CjgA6KJ9dOAIiItZK+DCzJcqdFxNpcPwk4HxgCXJ1LvT7MzKwJKk02EfEUsEuP2GMUT6f1LBvAyb20MweYUyPeDuxdI16zDzMzaw7PIGBmZpVzsjEzs8o52ZiZWeWcbMzMrHJONmZmVjknGzMzq5yTjZmZVc7JxszMKudkY2ZmlXOyMTOzyjnZmJlZ5ZxszMysck42ZmZWOScbMzOrnJONmZlVzsnGzMwq52RjZmaVc7IxM7PKOdmYmVnlnGzMzKxylSYbScMkXSLpHkl3S3qDpBGSFklaka/Ds6wkzZTUIel2SfuX2pma5VdImlqKHyDpjqwzU5IyXrMPMzNrjqrPbL4JXBMRrwX2Ae4GpgOLI2ICsDi3AY4EJuQyDZgFReIAZgAHAQcCM0rJY1aW7a43KeO99WFmZk1QWbKRNBR4M3AuQET8OSIeByYDc7PYXODoXJ8MzIvCzcAwSaOAI4BFEbE2ItYBi4BJuW9oRNwUEQHM69FWrT7MzKwJqjyz2QPoAs6TdJuk70naEdg9IlYD5OtuWX40sKpUvzNj9eKdNeLU6eN5JE2T1C6pvaurq//v1MzM6qoy2QwG9gdmRcR+wB+ofzlLNWLRj3jDImJ2RLRFRNvIkSM3paqZmW2CKpNNJ9AZEbfk9iUUyeeRvARGvq4plR9bqj8GeKiP+Jgacer0YWZmTVBZsomIh4FVkl6ToUOBu4D5QPcTZVOBy3N9PjAln0qbCKzPS2ALgcMlDc8HAw4HFua+DZIm5lNoU3q0VasPMzNrgsEVt/9R4IeStgPuA06gSHAXSToReBA4JsteBRwFdABPZVkiYq2kLwNLstxpEbE2108CzgeGAFfnAnB6L32YmVkTVJpsImIp0FZj16E1ygZwci/tzAHm1Ii3A3vXiD9Wqw8zM2sOzyBgZmaVc7IxM7PKOdmYmVnlnGzMzKxyTjZmZlY5JxszM6uck42ZmVXOycbMzCrnZGNmZpVzsjEzs8o52ZiZWeWcbMzMrHJONmZmVjknGzMzq5yTjZmZVc7JxszMKudkY2ZmlXOyMTOzyjnZmJlZ5ZxszMyscpUmG0krJd0haamk9oyNkLRI0op8HZ5xSZopqUPS7ZL2L7UzNcuvkDS1FD8g2+/IuqrXh5mZNcdAnNm8JSL2jYi23J4OLI6ICcDi3AY4EpiQyzRgFhSJA5gBHAQcCMwoJY9ZWba73qQ++jAzsyZoxmW0ycDcXJ8LHF2Kz4vCzcAwSaOAI4BFEbE2ItYBi4BJuW9oRNwUEQHM69FWrT7MzKwJqk42AVwr6VZJ0zK2e0SsBsjX3TI+GlhVqtuZsXrxzhrxen08j6RpktoltXd1dfXzLZqZWV8GV9z+wRHxkKTdgEWS7qlTVjVi0Y94wyJiNjAboK2tbZPqmplZ4yo9s4mIh/J1DfBTinsuj+QlMPJ1TRbvBMaWqo8BHuojPqZGnDp9mJlZE1SWbCTtKGnn7nXgcGAZMB/ofqJsKnB5rs8HpuRTaROB9XkJbCFwuKTh+WDA4cDC3LdB0sR8Cm1Kj7Zq9WFmZk1Q5WW03YGf5tPIg4EfRcQ1kpYAF0k6EXgQOCbLXwUcBXQATwEnAETEWklfBpZkudMiYm2unwScDwwBrs4F4PRe+jAzsyaoLNlExH3APjXijwGH1ogHcHIvbc0B5tSItwN7N9qHmZk1h2cQMDOzylX9NNpWbdz0BTXjK09/+wCPxMysuXxmY2ZmlXOyMTOzyvWZbCS9StL2uX6IpI9JGlb90MzMrFU0cmbzE+AZSXsC5wLjgR9VOiozM2spjSSbZyNiI/Bu4OyI+AQwqtphmZlZK2kk2fxF0nEU38S/MmPbVjckMzNrNY0kmxOANwBfjYj7JY0HflDtsMzMrJX0+T2biLhL0meBV+T2/RTTwZiZmTWkkafR3gksBa7J7X0lza96YGZm1joauYz2JYqfBngcICKWUjyRZmZm1pBGks3GiFjfI+YfGjMzs4Y1MjfaMkn/BAySNAH4GHBjtcMyM7NW0siZzUeBvwOeBi4AngA+XuWgzMystTTyNNpTwBdyMTMz22S9JhtJV1Dn3kxEvKuSEZmZWcupd2bz9QEbhZmZtbRek01E/Kx7XdJ2wGspznSWR8SfB2BsZmbWIhr5UufbgXuBmcC3gA5JRzbagaRBkm6TdGVuj5d0i6QVkn6ciQxJ2+d2R+4fV2rjcxlfLumIUnxSxjokTS/Fa/ZhZmbN0cjTaN8A3hIRh0TEPwBvAc7ahD5OBe4ubX8NOCsiJgDrgBMzfiKwLiL2zPa/BiBpL+BYiifiJgHfzgQ2CDgHOBLYCzguy9brw8zMmqCRZLMmIjpK2/cBaxppXNIY4O3A93JbwFuBS7LIXODoXJ+c2+T+Q7P8ZODCiHg652XroJjR4ECgIyLuy8t6FwKT++jDzMyaoJEvdd4p6SrgIop7NscASyS9ByAiLq1T92zgM8DOub0L8Hj+Pg5AJzA610cDq7LNjZLWZ/nRwM2lNst1VvWIH9RHH2Zm1gSNnNnsADwC/ANwCNAFjADeCbyjt0qS3kFxVnRrOVyjaPSx78WK1xrjNEntktq7urpqFTEzsxdBI1/qPKGfbR8MvEvSURQJayjFmc4wSYPzzGMM8FCW7wTGAp2SBgMvBdaW4t3KdWrFH63TR8/3NhuYDdDW1ub53szMKtLI02jjJZ0p6VJJ87uXvupFxOciYkxEjKO4wX9dRBwPXA+8L4tNBS7P9fm5Te6/LiIi48fm02rjgQnAr4ElwIQc33bZx/ys01sfZmbWBI3cs7kMOBe4Anj2Rejzs8CFkr4C3JZtk6/fl9RBcUZzLEBE3CnpIuAuYCNwckQ8AyDpFGAhMAiYExF39tGHmZk1QSPJ5k8RMfOFdBIRNwA35Pp9FE+S9SzzJ4qHD2rV/yrw1Rrxq4CrasRr9mFmZs3RSLL5pqQZwLUUMz8DEBG/qWxUZmbWUhpJNv8L+CDFd1e6L6NFbpuZmfWpkWTzbmAPz4dmZmb91cj3bH4LDKt6IGZm1roaObPZHbhH0hKef8/Gv2djZmYNaSTZzKh8FGZm1tIamUHgZ32VMTMzq6eRGQQmSloi6UlJf5b0jKQnBmJwZmbWGhp5QOBbwHHACmAI8M8ZMzMza0gj92yIiA5Jg3KamPMk3VjxuMzMrIU0kmyeyokul0o6A1gN7FjtsMzMrJU0chntg1nuFOAPFNP6v7fKQZmZWWtp5Gm0B3L1T5JmAmN7/Ey0mZlZXY08jXaDpKGSRlDMJnCepDOrH5qZmbWKRi6jvTQingDeA5wXEQcAh1U7LDMzayWNJJvBkkYB/whcWfF4zMysBTWSbE6j+DXMjohYImkPiu/cmJmZNaSRBwQuBi4ubd+Hn0YzM7NN0MiZjZmZ2QviZGNmZpWrLNlI2kHSryX9VtKdkv4j4+Ml3SJphaQf5+wESNo+tzty/7hSW5/L+HJJR5TikzLWIWl6KV6zDzMza45Gvmfzb6X17Teh7aeBt0bEPsC+wCRJE4GvAWdFxARgHXBilj8RWBcRewJnZTkk7QUcC/wdMAn4tqRBkgYB5wBHAnsBx2VZ6vRhZmZN0GuykfQZSW8A3lcK39Row1F4Mje3zSWAtwKXZHwucHSuT85tcv+hkpTxCyPi6Yi4H+gADsylIyLui4g/AxcCk7NOb32YmVkT1DuzWQ4cA+wh6ReSZgO7SHpNo43nGchSYA2wCLgXeDwiNmaRTmB0ro8GVgHk/vXALuV4jzq9xXep00fP8U2T1C6pvaurq9G3ZWZmm6heslkHfJ7iTOIQYGbGpzf6EwMR8UxE7AuMoTgTeV2tYvmqXva9WPFa45sdEW0R0TZy5MhaRczM7EVQL9lMAhYArwLOpEgWf4iIEyLijZvSSUQ8DtwATASGSer+fs8Y4KFc76SYUZrc/1JgbTneo05v8Ufr9GFmZk3Qa7KJiM9HxKHASuAHFF8AHSnpl5Ku6KthSSMlDcv1IRTzqd0NXM9z94GmApfn+vzcJvdfFxGR8WPzabXxwATg18ASYEI+ebYdxUME87NOb32YmVkTNPLjaQsjYgmwRNJJEfEmSbs2UG8UMDefGtsGuCgirpR0F3ChpK8AtwHnZvlzge9L6qA4ozkWICLulHQRcBewETg5fzEUSadQTKUzCJgTEXdmW5/tpQ8zM2sCFScCDRaW9omI31Y4nqZpa2uL9vb2ftUdN33BJpVfefrb+9WPmdnmRtKtEdHWV7lN+lJnqyYaMzOrlqerMTOzyjnZmJlZ5ZxszMysck42ZmZWOScbMzOrnJONmZlVzsnGzMwq52RjZmaVc7IxM7PKOdmYmVnlnGzMzKxyTjZmZlY5JxszM6uck42ZmVXOycbMzCrnZGNmZpVzsjEzs8o52ZiZWeWcbMzMrHKVJRtJYyVdL+luSXdKOjXjIyQtkrQiX4dnXJJmSuqQdLuk/UttTc3yKyRNLcUPkHRH1pkpSfX6MDOz5qjyzGYj8KmIeB0wEThZ0l7AdGBxREwAFuc2wJHAhFymAbOgSBzADOAg4EBgRil5zMqy3fUmZby3PszMrAkqSzYRsToifpPrG4C7gdHAZGBuFpsLHJ3rk4F5UbgZGCZpFHAEsCgi1kbEOmARMCn3DY2ImyIigHk92qrVh5mZNcGA3LORNA7YD7gF2D0iVkORkIDdsthoYFWpWmfG6sU7a8Sp00fPcU2T1C6pvaurq79vz8zM+lB5spG0E/AT4OMR8US9ojVi0Y94wyJidkS0RUTbyJEjN6WqmZltgkqTjaRtKRLNDyPi0gw/kpfAyNc1Ge8ExpaqjwEe6iM+pka8Xh9mZtYEVT6NJuBc4O6IOLO0az7Q/UTZVODyUnxKPpU2EVifl8AWAodLGp4PBhwOLMx9GyRNzL6m9GirVh9mZtYEgyts+2Dgg8AdkpZm7PPA6cBFkk4EHgSOyX1XAUcBHcBTwAkAEbFW0peBJVnutIhYm+snAecDQ4Crc6FOH2Zm1gSVJZuI+CW176sAHFqjfAAn99LWHGBOjXg7sHeN+GO1+jAzs+bwDAJmZlY5JxszM6uck42ZmVXOycbMzCrnZGNmZpVzsjEzs8o52ZiZWeWcbMzMrHJONmZmVjknGzMzq5yTjZmZVc7JxszMKudkY2ZmlXOyMTOzyjnZmJlZ5ZxszMysck42ZmZWOScbMzOrnJONmZlVzsnGzMwqV1mykTRH0hpJy0qxEZIWSVqRr8MzLkkzJXVIul3S/qU6U7P8CklTS/EDJN2RdWZKUr0+zMyseao8szkfmNQjNh1YHBETgMW5DXAkMCGXacAsKBIHMAM4CDgQmFFKHrOybHe9SX30YWZmTVJZsomInwNre4QnA3NzfS5wdCk+Lwo3A8MkjQKOABZFxNqIWAcsAiblvqERcVNEBDCvR1u1+jAzsyYZ6Hs2u0fEaoB83S3jo4FVpXKdGasX76wRr9fH35A0TVK7pPaurq5+vykzM6tvc3lAQDVi0Y/4JomI2RHRFhFtI0eO3NTqZmbWoIFONo/kJTDydU3GO4GxpXJjgIf6iI+pEa/Xh5mZNclAJ5v5QPcTZVOBy0vxKflU2kRgfV4CWwgcLml4PhhwOLAw922QNDGfQpvSo61afZiZWZMMrqphSRcAhwC7SuqkeKrsdOAiSScCDwLHZPGrgKOADuAp4ASAiFgr6cvAkix3WkR0P3RwEsUTb0OAq3OhTh9mZtYklSWbiDiul12H1igbwMm9tDMHmFMj3g7sXSP+WK0+zMyseTaXBwTMzKyFOdmYmVnlnGzMzKxyTjZmZlY5JxszM6uck42ZmVXOycbMzCrnZGNmZpVzsjEzs8o52ZiZWeUqm67Gejdu+oKa8ZWnv32AR2JmNjB8ZmNmZpVzsjEzs8o52ZiZWeWcbMzMrHJONmZmVjknGzMzq5yTjZmZVc7JxszMKudkY2ZmlWvZZCNpkqTlkjokTW/2eMzMtmYtmWwkDQLOAY4E9gKOk7RXc0dlZrb1atW50Q4EOiLiPgBJFwKTgbuaOqo+eM40M2tVrZpsRgOrStudwEE9C0maBkzLzSclLe9nf7sCj/azbp/0tapaHlCVHqMW4WPUGB+nvg3kMXplI4VaNdmoRiz+JhAxG5j9gjuT2iOi7YW208p8jPrmY9QYH6e+bY7HqCXv2VCcyYwtbY8BHmrSWMzMtnqtmmyWABMkjZe0HXAsML/JYzIz22q15GW0iNgo6RRgITAImBMRd1bY5Qu+FLcV8DHqm49RY3yc+rbZHSNF/M2tDDMzsxdVq15GMzOzzYiTjZmZVc7J5gXwlDgFSWMlXS/pbkl3Sjo14yMkLZK0Il+HZ1ySZuZxu13S/s19BwNH0iBJt0m6MrfHS7olj9GP84EWJG2f2x25f1wzxz2QJA2TdImke/Iz9QZ/lp5P0ify79oySRdI2mFz/yw52fSTp8R5no3ApyLidcBE4OQ8FtOBxRExAVic21Acswm5TANmDfyQm+ZU4O7S9teAs/IYrQNOzPiJwLqI2BM4K8ttLb4JXBMRrwX2oThe/iwlSaOBjwFtEbE3xUNQx7K5f5Yiwks/FuANwMLS9ueAzzV7XJvDAlwOvA1YDozK2Chgea7/F3Bcqfxfy7XyQvF9r8XAW4ErKb58/CgwuOdniuJJyjfk+uAsp2a/hwE4RkOB+3u+V3+WnncsumdIGZGfjSuBIzb3z5LPbPqv1pQ4o5s0ls1GnqLvB9wC7B4RqwHydbcstrUeu7OBzwDP5vYuwOMRsTG3y8fhr8co96/P8q1uD6ALOC8vN35P0o74s/RXEfF74OvAg8Bqis/GrWzmnyUnm/5raEqcrYmknYCfAB+PiCfqFa0Ra+ljJ+kdwJqIuLUcrlE0GtjXygYD+wOzImI/4A88d8mslq3uOOX9qsnAeODlwI4UlxN72qw+S042/ecpcUokbUuRaH4YEZdm+BFJo3L/KGBNxrfGY3cw8C5JK4ELKS6lnQ0Mk9T95erycfjrMcr9LwXWDuSAm6QT6IyIW3L7Eork48/Scw4D7o+Iroj4C3Ap8EY288+Sk03/eUqcJEnAucDdEXFmadd8YGquT6W4l9Mdn5JPEk0E1ndfImlVEfG5iBgTEeMoPivXRcTxwPXA+7JYz2PUfezel+Vb+n/sABHxMLBK0msydCjFT4P4s/ScB4GJkl6Sf/e6j9Hm/Vlq9s2uLXkBjgJ+B9wLfKHZ42nicXgTxWn57cDSXI6iuC68GFiRryOyvCie5LsXuIPiqZqmv48BPF6HAFfm+h7Ar4EO4GJg+4zvkNsduX+PZo97AI/PvkB7fp4uA4b7s/Q3x+g/gHuAZcD3ge0398+Sp6sxM7PK+TKamZlVzsnGzMwq52RjZmaVc7IxM7PKOdmYmVnlnGysJUh6soI2Jek6SUNf7LZ79HODpLYq+8h+PpazKP+wR3xfSUc1UP9Lkv71RRjHSEnXvNB2bMviZGPWu6OA30b9qXeaqvSN8Ub8C3BUFF8mLduX4r0OiIjoAlZLOnig+rTmc7KxlpX/g/6JpCW5HJzxL0mak2cU90n6WC9NHE9+C1vSuDwr+G7+jsi1kobkvr+emUjaNaekQdKHJF0m6QpJ90s6RdInc4LJmyWNKPX1AUk35u+THJj1d8xxLsk6k0vtXizpCuDaGu/7k9nOMkkfz9h3KL70N1/SJ0pltwNOA94vaamk96v47ZjLVPw+zM2S/r5GH/9X0tWShkh6laRrJN0q6ReSXptlzlfxWzM35nF+X6mJy/L42tai2d+E9eLlxViAJ2vEfgS8KddfQTGdDsCXgBspvnW9K/AYsG2N+g8AO+f6OIrf7dk3ty8CPpDrN5DfXM/2Vub6hyi+tb0zMJJitt2P5L6zKCYs7a7/3Vx/M7As1/9fqY9hFLNV7JjtdpLfou8x5gMovkm/I7ATcCewX+5bCexao86HgG+Vtv8TmJHrbwWWlo7bvwKnUEyB0v0N9cXAhFw/iGI6FIDzKb65vg3Fbz51lPoYDdzR7M+Nl4FbNuUU3GxLcxiwVzF9FABDJe2c6wsi4mngaUlrgN0p/gEvGxERG0rb90fE0ly/lSIB9eX6bGODpPXAFRm/AyifMVwAEBE/lzRU0jDgcIrJO7vvk+xAkTQBFkVErckU3wT8NCL+ACDpUuB/A7c1MNZyG+/N8VwnaRdJL819H6Q4TkdHxF9UzPT9RuDi0nHevtTWZRHxLHCXpN1L8TUUMxbbVsLJxlrZNhQ/GvXHcjD/UXy6FHqG2n9DVdLVAAABnklEQVQXNkraJv+xrFVnSHc5nrskvUOPNsp1ni1tP9ujz57zRgXFvF/vjYjlPcZ/EMXU+7XUmk5+U9Wbkn4ZxT2eMRQ/crYNxe+o7NtLW+X3X253B+CP2FbD92yslV1LcckHKJ662sT6yynuc/RlJcXlK3hu1t1N9X4ASW+imLl4PcUvLH40Z/ZF0n4NtPNz4OicEXhH4N3AL/qos4HiUl+5jeOzz0OAR+O5hyRuAz5Mce/n5Rm/X9IxWV6S9mlgnK+mSFy2lXCysVbxEkmdpeWT5O+0543uu4CPbGKbCyhmaO7L14GTJN1Icc+mP9Zl/e/w3G/HfxnYFrhd0rLcrisifkNxr+TXFL+W+r2I6OsS2vUUlxuXSno/xb2ZNkm3A6fz3PT03X38kuLezQJJu1IkphMl/ZbiHtHkvt8ub6E4vraV8KzPZr1Q8SNd8yLibc0eS6uR9HNgckSsa/ZYbGD4zMasF1H8CNd3q/5S59ZG0kjgTCearYvPbMzMrHI+szEzs8o52ZiZWeWcbMzMrHJONmZmVjknGzMzq9z/ALxomvsa4LkdAAAAAElFTkSuQmCC )

**Tokenization**

```python
First sample before preprocessing: 
 <NAME/> actually if I were closer I'd stop by for some of your gluten free pancakes! (with chocolate ice cream of course) 

First sample after preprocessing: 
 [    1   292    78     2   171  1893   401   339   121    12    66    13
    48 10960   375  2111    22   727   588   666    13   544]
```



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
Epoch 00001: val_acc improved from -inf to 0.80551, saving model to models/weights-improvement-01-0.81.hdf5

Epoch 00002: val_acc improved from 0.80551 to 0.81091, saving model to models/weights-improvement-02-0.81.hdf5
310600/310600 [==============================] - 11s 34us/step

Accuracy:  81.0592401802962
```

**Confusion Matrix**:

![img](data:image/png;base64,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 )

```python
  precision    recall  f1-score   support

   negative       0.80      0.82      0.81    152784
   positive       0.82      0.81      0.81    157816

avg / total       0.81      0.81      0.81    310600
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

