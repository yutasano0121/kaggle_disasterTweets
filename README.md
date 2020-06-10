# Codes for a kaggle competition "Real or Not? NLP with Disaster Tweets"

## Purpose
To predict whether a given tweet ('text') is about a real disaster/threat ('target' = 1) or not ('target' = 0).<br><br>
Data available at https://www.kaggle.com/c/nlp-getting-started/

## Folders
`./` contains `.ipynb` files for analyses. Raw data (`train.csv` and `test.csv`) should also be plaved in this folder. <br>
<br>
`source` contains scripts for training and serving TensorFlow models.<br>
<br>
`processed_data` and `cache` folders will be created when the notebooks are run.

## Workflow
* **XGBoost model**<br>
    1. Run `dataExploration.ipynb`, which converts text into bag of words.
    2. Run `XGBoost_train.ipynb`, which train a model with hyperparameter tuning.
    3. Run `XGBoost_test.ipynb`, which make prediction using the trained model.<br>
<br>

F1 score: 0.67280

* **TensorFlow model**<br>
    1. Run `dataExploration.ipynb`.
    2. Run `tensorflow_train.ipynb`, which fits a very simple TensorFlow model.
    3. Run `tensorflow_test.ipynb`.<br>
<br>

F1 score: 0.69836 (Another model **without** tuning and early stopping had a better score, 0.72801. Maybe due to the early stopping?)

* **Bert model**<br>
    Adopted from this shared notebook https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert for an implementation in SageMaker. <br>
    Just run `bert_train.ipynb` and all the analyses (data exploration/training/prediction) will be done.<br>

    Changes from the original notebook:<br>
    * Data preprocessing was simplified from manual relabeling to just removing stopwords and punctuations using `nltk` and `re`.
    * Mislabeled tweets were removed from the data instead of being relabeled.
    * Duplicated tweets were removed.
    * Cross validation was removed.
    * Epochs were increased to 20 with early stopping.<br>

F1 score: 0.79038<br>
<br>
*Scores can be obtained by submitting the prediction at Kaggle, or matching it with data here https://www.kaggle.com/jannesklaas/disasters-on-social-media?select=socialmedia-disaster-tweets-DFE.csv.*
