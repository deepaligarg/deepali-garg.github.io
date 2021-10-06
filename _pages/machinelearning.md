---

permalink: /data-wrangling/
title: "Data Science Projects"
author_profile: true
header:
  image: "/images/pic_project_crop.jpg"
---


---
## Natural Language Processing with Deep Learning

### Sentiment Analysis Using IMDB Movies Reviews

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews/blob/master/IMDB%20movies%20review%20sentiment%20analysis.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

<span style="font-family:Segoe UI; font-size:1;">This is an NLP and Flask based application which involves predicting the sentiments of the sentences as positive or negative. The classifier is trained on a huge dataset of IMDB movies reviews.  The model is then hosted using Flask to be used by end users.
This project has text pre-processing done through NLTK and Regex and EDA for understanding the features and data well.
The text is then coverted into vectors using 2 techniques - **Countvectorize and TF-IDF**.
Two Machine Learning algorithms **(Naive Bayes and SVM)** are then used with combonitions of above 2 techniques and it is found that Naive Bayes with TF-IDF outstands the other algorithm.
The model is then saved in a **Pickle file** and used in the **Flask Application** to host the website on localhost.</span>

---
### Spotify-EDA

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Spotify-EDA)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Spotify-EDA/blob/master/Spotify%20Popularity%20and%20EDA_ANN.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/mrmorj/dataset-of-songs-in-spotify)

<span style="font-family:Segoe UI; font-size:1;">Spotify is an extremely popular music application. This project consumes a large dataset of songs and gives various reports and trends on those songs.
The code contains text pre-processing and detailed EDA to yield following trends : </span>

 1. 20 Most Popular Songs.
 2. 20 Most Popular Artists.
 3. Audio Characteristics over the year.
 4. Artists with maximum number of Songs.
 5. Songs released Year-wise. 

<span style="font-family:Segoe UI; font-size:1;">Also the popularity of the songs have been estimated using Deep Learning with **Neaural Networks**. </span>

 
 ---
### Classifying Fake News

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Fake-News-Classification-Deep-Learning)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Fake-News-Classification-Deep-Learning/blob/master/Fake%20News%20Classifier%20with%20LSTM.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/c/fake-news)
 
<span style="font-family:Segoe UI; font-size:1;">A Deep Learning approach of classifying the news headlines and its content as Fake or Real.
A considerable amount of labeled news headlines and content are taken and a Deep Learning approach is used to classify any news as Fake or Real.
Text pre-processing is done using **NLTK** library. The words are converted into vectors using **Word Embeddings**.
The model is built using **LSTM** and **Bi-directional LSTM** with **Dropout Layers**.
It was found that LSTM out-performed Bi-diectional LSTM for this use-case.</span>

---
### Spam Classifier

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/-Spam-Classifier)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/-Spam-Classifier/blob/master/Spam%20Classification.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

<span style="font-family:Segoe UI; font-size:1;">This NLP project reads a message and classifies them as Spam or Ham (Not Spam). This uses NLTK for text preprocessing and Machine learning algorithms for classifying text messages.
Text pre-processing done by applying **Regex**, **Stemming** and Removing **Stopwords**. 
The words are then converted into words using **Bag Of Words** Technique (Sklearn's **CountVecrtorizer**) and then a **Naive Bayes Classifier** is built for the use-case.</span>

---
### Digit Recognizer

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Digit-Recognizer)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Digit-Recognizer/blob/master/Digit%20Recognizer%20using%20ANN%20and%20Keras%20Hyperparameter%20Tuning.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/c/digit-recognizer)

<span style="font-family:Segoe UI; font-size:1;">This application recognizes hand-written digits from 0-9 and labels them correctly on basis of certain characteristics. This is an example of using Neural Networks and Deep Learning.
This notebook depicts the use of **Deep Learning** technique - **Artificial Neural Networks**. The networks are built using **Keras** Library and hyperparameter tuning is also performed to find best model parameters for this use-case using **Keras Tuner's Random Search**. The the model's performance is analysed using Confusion Matrix and Classification Report.</span>

---
## Unsupervised Machine Learning

### Clustering of Retail Customers

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Clustering-Retail-Customers)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Clustering-Retail-Customers/blob/master/RetailDS_clustering.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/hellbuoy/online-retail-customer-clustering)

<span style="font-family:Segoe UI; font-size:1;">This application is for clustering and grouping customers based on their purchase history to give insights to the retail department.
For clustering, unsupervised machine learning algorithms are used such as- </span>

1. KMeans-Clustering
2. Silhoutte Method for determining optimal values of clusters(K in KMeans)
3. Hierarchial Clustering - Methods: Single, Average and Complete.
4. DBSCAN 


---
### Clustering of Credit Card Users

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Clustering-Credit-Card-Users)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Clustering-Credit-Card-Users/blob/master/CreditDS_clustering_UnsupML.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/arjunbhasin2013/ccdata)


<span style="font-family:Segoe UI; font-size:1;">This is a use-case of clustering the credit card customers on the basis of various attributes using Unsupervised Machine Learning Techniques.
The project is a blend of detailed EDA and unsupervised machine learning algorithms such as **K-Means Clustering, Hierarchial Clustering and DBSCAN.** </span>


---
## Supervised Machine Learning

### Handling Imbalanced Credit Card Dataset

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Credit-Card-Imbalanced-Dataset)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Credit-Card-Imbalanced-Dataset/blob/master/CreditDS_clustering_UnsupML.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/arjunbhasin2013/ccdata)

<span style="font-family:Segoe UI; font-size:1;">Mostly in Banking domains or credit card use cases, the data for predicting a transaction as fraudulent is extremely low due to less evidence for fraud cases resulting in an Imbalanced Dataset for ML use cases. This projrct deals with 3 techniques of handling such cases. </span>

The 3 Techniques discussed in the notebook are :  
1. Under-sampling
2. Over-sampling
3. SMOTE Technique

Then a **Random Forest** algorithm is applied to check the performance of each technique.


---
### Wine Quality Prediction

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Wine-Quality-Prediction-)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Wine-Quality-Prediction-/blob/master/WineQuality_DT%20and%20Ensemble.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)

<span style="font-family:Segoe UI; font-size:1;">This is a classic classification example of predicting and classifying Wine Quality on a scale of 3 to 8. The lowest scale being the poor quality and the highest scale being the best quality.
The project is an amalgamation of detailed EDA and supervised Ensemble Techniques such has **Random Forest** and **Gradient Boost**. The values are also scaled using Standard Scaler and Classification Report is generated at the end to analyse the prediction and classification done.</span>


---
### Breast Cancer Prediction

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Breast-Cancer-Prediction)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Breast-Cancer-Prediction/blob/master/BreastCancerPred_LogRegr.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/c/breast-cancer-classification)

<span style="font-family:Segoe UI; font-size:1;">This is an analysis of predicting cancer chances in the patients considering different parameters of the human cells.
The project contains detailed EDA of the dataset taken and the prediction is done by using Logistic Regression and KNN Algorithms. Classification Report with Accuracy and F1 score are also analysed with using these algorithms.</span>


---
### Cars Price Prediction

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Cars-Price-Prediction)
[![View on GitHub](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Cars-Price-Prediction/blob/master/DataExplore_carspriceprediction_LinReg.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho)


<span style="font-family:Segoe UI; font-size:1;">This project involves predicting the prices of the cars in order to give insights to the company to set up business in different locations.
The project containes EDA and VIF (Variance Inflation Factor) for measuring amount of multicolinearity in a set of multiple regression variables. 
Feature seclection is done through sklearn's library - RFE and prediction by applying Linear Regression.</span>


---
### Salary Prediction

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Salary-Prediction)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Salary-Prediction/blob/master/PredictingSalaries_LinearReg.ipynb)
[![View Dataset](https://img.shields.io/badge/Dataset-View_Dataset-blueviolet?logo=Microsoft%20Excel)](https://www.kaggle.com/laleeth/salary-predict-dataset)

<span style="font-family:Segoe UI; font-size:1;">This preoject contains a Python notebook that helps in analyzing the Salary trends and predicting Salaries on the basis of Years of Experience.
A simple Linear Regression Algorithm is applied and the score is calculated which gives an **accuracy of 96%**.</span>



 





