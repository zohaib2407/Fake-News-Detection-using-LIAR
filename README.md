# Fake-News-Detection-using-LIAR
Aim to perform binary classification of our Liar data with the help of concepts pertaining to Natural Language Processing and Machine Learning. 

IDS 566 – Advanced Text Analytics (SP’22)
Final Project Report – “Fake News Detection” - Group 20
Kush Mashru, Yang Yang, Zohaib Sheikh

## Motivation
Fake news has evolved into a serious problem for society as well as a significant task for those attempting to combat disinformation. This problem has harmed democratic elections, individual people, or organizations' reputations, and has had a negative impact on citizens (e.g., during the COVID-19 pandemic in the US). The propagation of fake news has far-reaching repercussions, from altering election outcomes in favor of specific politicians to creating prejudiced viewpoints. 

## Objective
We aim to perform binary classification of our Liar data with the help of concepts pertaining to Natural Language Processing and Machine Learning. We aim to provide the user with the ability to classify the news as fake or real.

## Dataset 
The LIAR dataset consists of 12,836 brief statements extracted from POLITIFACT and manually classified for honesty, subject, context/venue, speaker, state, party, and history by humans. The LIAR dataset comprises six categories for truthfulness: pants-on-fire, false, mostly false, half-true, mostly-true, and true. The sizes of these six label sets are rather evenly distributed. The statements were gathered from a variety of media, including TV appearances, speeches, tweets, and debates, and they span a wide range of themes, including the economy, health care, taxes, and the election.
Evidence sentences were mechanically extracted from the full-text judgment report submitted by journalists in PolitiFact for this dataset. Our goal is to establish a standard for evidence retrieval and demonstrate that integrating evidence information in any automatic fake news detection approach (independent of features or classifier) always outperforms methods that do not include it.

## Exploratory Data Analysis 
The original data was divided into three datasets: Training set size (10,269); Validation set size (1,284); Testing set size (1,283). For our analysis purpose we append all three datasets to form a new dataset and then split those datasets into training and testing data. Also, when we see the distribution of speaker affiliations, we can see that almost 35% of the speakers are affiliated to Democrats while 47% to the Republic. Remaining 18% to others. (e.g., FB posts) 

## Pre-processing 
We concentrated our data processing efforts on the text column, which carries the news content. To make the model more predictable, we changed this text column to extract additional information. We used a package called 'NLTK' to extract information from the text column.
We utilized the 'NLTK' library to removing stop words, perform Tokenization, Stemming, and Lemmatization functions here. So, with these three instances, we employed each of these functions one by one. We decided that stemming would be a better match for this data, thus we dropped the data's lemmatization.

## Preparing two datasets for training different models  
Over here we are training model on two different datasets:
a)	TFIDF Vectorization 
The frequency with which words appear in a document is referred to as term frequency. The IDF is a measure of how important a phrase is over the whole corpus. Collection of word Documents were converted into the matrix which contains TF-IDF features using  TfidfVectorizer.

b)	TFIDF + Topic Modeling
In addition to the TFIDF vectorization we intended to extract some more information from the news statements. Hence, we chose to perform topic modeling and use topics as one of the features for our ML model. After cross validation and manual selection, we converged to a total of 10 topics. For each statement, we have the proportion of how much is the contribution of the 10 topic is.

PCA to retain 95% of variations 
Over here we are applying PCA to retain 95% of variation in our datasets, before training Logistic Regression, XG Boost, Random Forest. We notice that the accuracy of the Naïve Bayes and SVM model reduces drastically, and runtime increases a lot. We decided to run those models without applying PCA for data reductions
Data Set	Train Dataset size	After applying PCA Train dataset size
TF IDF Data	(8953, 8244)	(8953,5204)
TFIDF + Topic Modeling	(8953, 8254)	(8953, 5204)

## Machine Learning Methodologies
For this classification task, the following models along with their parameters were selected:
1.	Logistic Regression - Log Loss, elastic net penalty, l1 ratio = 0.05, reg param – 0.01
2.	XG Boost - stopping – num of trees = 200, max_depth = 15, regularization parameter – 0.1
3.	Random Forest - Hyperparameter tuning – using max_depth and  number of estimators
4.	Naive Bayes - with 5-fold cross validation
5.	SVM – using linear kernel - with 5-fold cross validation

## Model Evaluation
Dataset	Models	Accuracy 	Precision	Recall	F1-score <br>
TF-IDF	Logistic Classifier	0.566	0.55	0.57	0.56 <br>
TF-IDF + Topics		0.537	0.53	0.54	0.54<br>
TF-IDF	XGBoost	0.57	0.55	0.57	0.55<br>
TF-IDF + Topics		0.58	0.56	0.59	0.55<br>
TF-IDF	Random Forest	0.58	0.55	0.59	0.49<br>
TF-IDF + Topics		0.59	0.57	0.59	0.47<br>
TF-IDF	Naïve Bayes	0.59	0.56	0.59	0.48<br>
TF-IDF + Topics		0.59	0.56	0.59	0.47<br>
TF-IDF	SVM	0.52	0.53	0.53	0.53<br>
TF-IDF + Topics		0.53	0.54	0.53	0.53<br>

## Evaluation 
As we can see that our best performing model is Random Forest with TF-IDF + Topics with just 59% accuracy. The reason of such a low accuracy is that LIAR dataset is an artificial dataset curated for research and training purpose and is infamous for not giving good results with tradition machine learning models. For future improvements, we can introduce some new feature engineering techniques such as POS tagging, NER, Sentiments as features, Word2Vec embeddings and use BERT Model.

