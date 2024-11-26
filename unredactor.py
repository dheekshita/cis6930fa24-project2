import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from pathlib import Path

def load_reviews(review_dir):
    reviews = []
    labels = []

    for folder in ['pos', 'neg']:
        folder_path = Path(review_dir) / folder
        for file in folder_path.iterdir():
            if file.is_file():
                with file.open(encoding='utf-8') as f:
                    reviews.append(f.read())
                labels.append(1 if folder =='pos' else 0)
    review_data = pd.DataFrame({'context': reviews, 'label': labels})
    return review_data


def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', names=['split', 'name', 'context'], on_bad_lines='skip')
    return data

def preprocess_context(context):
    return re.sub(r'█+', '[REDACTED]', context)

def combined_vectorizer(reviews, unredactor_data):
    combined_contexts = pd.concat([reviews['context'], unredactor_data['context']], ignore_index=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
    vectorizer.fit(combined_contexts)
    return vectorizer


def train_model(X, y):
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X, y)
    return model

def evaluate_model(model, vectorizer, validation_data):
    X_val = vectorizer.transform(validation_data['context'])
    y_true = validation_data['name']
    y_pred = model.predict(X_val)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, zero_division=1, average='weighted'
    )
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return y_pred

def predict_test(model, vectorizer, test_file, output_file):
    test_data = pd.read_csv(test_file, sep='\t', names=['id', 'context'])
    test_data['context'] = test_data['context'].apply(preprocess_context)
    X_test = vectorizer.transform(test_data['context'])
    test_data['name'] = model.predict(X_test)
    test_data[['id', 'name']].to_csv(output_file, sep='\t', index=False)
    print(f"Predictions saved to {output_file}")

def main():
    training_file = 'unredactor.tsv'
    review_dir = 'aclImdb/train'
    test_file = 'test.tsv'
    output_file = 'submission.tsv'

    unredactor_data = load_data(training_file)
    unredactor_data['context'] = unredactor_data['context'].apply(preprocess_context)

    train_data = unredactor_data[unredactor_data['split'] == 'training']
    validation_data = unredactor_data[unredactor_data['split'] == 'validation']

    reviews = load_reviews(review_dir)
    print(f"Loaded {len(reviews)} movie reviews.")

    vectorizer = combined_vectorizer(reviews, unredactor_data)

    X_train = vectorizer.transform(train_data['context'])
    y_train = train_data['name']

    model = train_model(X_train, y_train)

    print("Evaluating on validation set:")
    evaluate_model(model, vectorizer, validation_data)

    print("Predicting on test set:")
    predict_test(model, vectorizer, test_file, output_file)

if __name__ == '__main__':
    main()
