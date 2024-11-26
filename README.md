# cis6930fa24 -- Project 2

Name: Dheekshita Neella

# Project Description
This project is a model that predicts redacted names in a text. It first take the training data, preprocess it, train the model using Logistic Regression, and evaluate it's performance on validation dataset. This model is also trained with the movie reviews from the IMDB dataset. This model lastly predicts the redacted values of a test data and saves the predicted names into submission.tsv. 


# How to install
pipenv install

## How to run
python unredactor.py

## Pipeline Instructions
1. Loads and preprocess movie reviews and redacted data from unredactor.tsv
2. Combines the datasets to fir a Tfidf vectorizer.
3. Trains a Logistic Regression model on both the movie review and training data from unredactor.tsv
4. Evaluates the model on the validation data from unredactor.tsv, and computes precision, recall, F1 score and accuracy.
5. Predicts names from the test.tsv set and saves the predictions to submission.tsv

## Functions
load_reviews(review_dir) - This function takes in the folder where the movie reviews are residing and reads the reviews and assigns label 1 to positive and 0 to negative reviews. Returns a dataframe with context and label

load_data(file_path) - This function takes the path to the unredactor.tsv file, reads the file with split, name and context columns, skipping the bad lines. Returns a dataframe with the file contents.

preprocess_context(context) - This function takes in the text containing redacted text, replaces the character '█' with [REDACTED], hence returns a clean text string.

combined_vectorizer(reviews, unredactor_data) - Takes in both the dataframes returned by load_reviews() and load_data(), combines the context from both of them to train a TFidf vectorizer. This function returns a fitted TFidf vectorizer, that would be useful for feature extraction.

train_model(x, y) - x is the vectorized training data, y contains labels for training, this function trains a logistic regression model with balanced class weights, and returns the trained model.

evaluate_model(model, vectorizer, validation_data) - takes in the trained logistic regression model, fitted vectorizer and validation dataframe, it predicts names for this data and calculates precision, recall, f1 score and accuracy for the same. Returns predicted labels.

predict_test(model, vectorizer, test_file, output_file) - takes in the trained model, fitted vectorizer, test.tsv path and path to save submission.tsv. It predicts the names from test.tsv and writes them into submission.tsv and saves it.



## Test cases
test_load_data.py - ensures the function load_data() properly reads a .tsv file and returns a dataframe, skipping bad lines.

test_preprocess_context.py - Tests that the function replaces the char '█' with [REDACTED]

test_combined_vectorizer.py - Tests if the functions fits a TFidf vectorizer with both movie reviews and unredactor contexts.

test_train_model.py - checks the training of a Logistic Regression model with given inputs.

test_predict_test.py - Ensures the function predicts the names from test.tsv and save them to submission.tsv properly.


## Bugs and Assumptions
1. The model might overfit if the vectorizer features align closely with the training dataset.
2. The test.tsv is assumed to be formatted correctly with id and context columns.
3. This model has a high precision and lower recall and F1 score, which indicates overfiting on training data. It predicts only a part if the true names correctly. This leads to generating the same name multiple times. 
4. The preprocess_context() may fail incase of multi lined context without proper cleanup.





