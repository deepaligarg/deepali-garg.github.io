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





