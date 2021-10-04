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
[![View on GitHub](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)
[![View on GitHub](https://img.shields.io/badge/Database-View_Database-blueviolet?logo=Microsoft%20Excel)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)

This is an NLP and Flask based application which involves predicting the sentiments of the sentences as positive or negative. The classifier is trained on a huge dataset of IMDB movies reviews.  The model is then hosted using Flask to be used by end users.
This project has text pre-processing done through NLTK and Regex and EDA for understanding the features and data well.
The text is then coverted into vectors using 2 techniques - **Countvectorize and TF-IDF**.
Two Machine Learning algorithms **(Naive Bayes and SVM)** are then used with combonitions of above 2 techniques and it is found that Naive Bayes with TF-IDF outstands the other algorithm.
The model is then saved in a **Pickle file** and used in the **Flask Application** to host the website on localhost.

---
### Spotify-EDA

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Spotify-EDA)
[![View on GitHub](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)
[![View on GitHub](https://img.shields.io/badge/Database-View_Database-blueviolet?logo=Microsoft%20Excel)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)

Spotify is an extremely popular music application. This project consumes a large dataset of songs and gives various reports and trends on those songs.
The code contains text pre-processing and detailed EDA to yield following trends :
 1. 20 Most Popular Songs.
 2. 20 Most Popular Artists.
 3. Audio Characteristics over the year.
 4. Artists with maximum number of Songs.
 5. Songs released Year-wise
 Also the popularity of the songs have been estimated using Deep Learning with **Neaural Networks**.
 
 ---
 ### Fake-News-Classification
 
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Fake-News-Classification-Deep-Learning)
[![View on GitHub](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)
[![View on GitHub](https://img.shields.io/badge/Database-View_Database-blueviolet?logo=Microsoft%20Excel)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)
 
A Deep Learning approach of classifying the news headlines and its content as Fake or Real.
A considerable amount of labeled news headlines and content are taken and a Deep Learning approach is used to classify any news as Fake or Real.
Text pre-processing is done using **NLTK** library. The words are converted into vectors using **Word Embeddings**.
The model is built using **LSTM** and **Bi-directional LSTM** with **Dropout Layers**.
It was found that LSTM out-performed Bi-diectional LSTM for this use-case.

---
### Clustering-Retail-Customers

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Clustering-Retail-Customers)
[![View on GitHub](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)
[![View on GitHub](https://img.shields.io/badge/Database-View_Database-blueviolet?logo=Microsoft%20Excel)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)

This application is for clustering and grouping customers based on their purchase history to give insights to the retail department.
For clustering, unsupervised machine learning algorithms are used such as-

1. KMeans-Clustering
2. Silhoutte Method for determining optimal values of clusters(K in KMeans)
3. Hierarchial Clustering - Methods: Single, Average and Complete.
4. DBSCAN 

---
## Machine Learning

# Credit-Card-Imbalanced-Dataset

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/deepaligarg/Credit-Card-Imbalanced-Dataset)
[![View on GitHub](https://img.shields.io/badge/Jupyter-Open_Notebook-green?logo=Jupyter)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)
[![View on GitHub](https://img.shields.io/badge/Database-View_Database-blueviolet?logo=Microsoft%20Excel)](https://github.com/deepaligarg/Sentiment-Analysis-using-IMDB-movies-reviews)

Mostly in Banking domains or credit card use cases, the data for predicting a transaction as fraudulent is extremely low due to less evidence for fraud cases resulting in an Imbalanced Dataset for ML use cases. This projrct deals with 3 techniques of handling such cases. 

The 3 Techniques discussed in the notebook are :
1. Under-sampling
2. Over-sampling
3. SMOTE Technique

Then a Random Forest algorithm is applied to check the performance of each technique.

---


 





