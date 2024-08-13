# SMS Spam Identification
"A Kaggle notebook for the classification of SMS messages into spam and ham using ML and deep learning models."   


![](https://paritkansal121.odoo.com/web/image/296-7c0804e2/dataset-cover.webp)

​
## Kaggle Notebook:
Link: https://www.kaggle.com/code/paritkansal/spam-identification

## Project Overview: 
The goal of this project is to develop a system that can accurately classify SMS messages as either "ham" (non-spam) or "spam" using various machine learning models. The dataset used is the SMS Spam Collection Dataset, which contains a set of SMS messages labeled as either spam or ham. 

### Dataset:

Source: SMS Spam Collection Dataset

### Project Steps:

#### Data Loading and Exploration:
Load and inspect the dataset.
Clean the data by removing unnecessary columns and handling missing values.

#### Data Preprocessing:
Convert text to lowercase.
Remove special characters, stopwords, and perform lemmatization to standardize the text.

#### Feature Extraction:
Use techniques like Word2Vec to convert text messages into numerical vectors suitable for machine learning models.

#### Model Training and Evaluation:
- **Approach 1**: Average the vectors to get a single vector and then use ML models such as Random Forest and XGBoost for messages.
- **Approach 2**: Use word embedding along with bidirectional LSTM to train and evaluate a Deep Learning model.
- **Approach 3**: Use pretrained Word2Vec along with bidirectional LSTM to capture the sequential nature of text data for better performance.

### Technologies and Libraries Used:

**Programming Language**: Python

**Libraries**: Pandas, Numpy, NLTK, Gensim, Scikit-learn, XGBoost, TensorFlow, Keras

### Outcome: 

The project successfully develops models that accurately classify SMS messages as spam or ham. The performance metrics for the best model are as follows:

- **Precision**: 0.9823
- **Recall**: 0.9639
- **F1 Score**: 0.9730
These results demonstrate the effectiveness of different machine learning techniques in text classification, with the best model achieving high precision, recall, and F1 score, indicating robust performance in identifying spam messages while minimizing false positives and false negatives.
