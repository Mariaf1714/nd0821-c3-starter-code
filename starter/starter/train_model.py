# Script to train machine learning model.

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data
from ml.model_slice import slice

# Add code to load in the data.

data_path = "../data/cleaned_census.csv"
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

pickle.dump(encoder, open("../model/encoder.pickle", "wb")) 
pickle.dump(lb, open("../model/lb.pickle", "wb")) 

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
            test, categorical_features=cat_features,
label="salary", training=False, encoder=encoder, lb=lb)

# Train and save a model.
model = train_model(X_train, y_train)
pickle.dump(model, open("../model/model.pickle", "wb")) 

# Compute model metrics on slices of the data
preds = inference(model, X_test)

# save metrics to txt file
precision, recall, fbeta = compute_model_metrics(y_test, preds)
with open("../model/model_metrics.txt", "a") as f:
    print(f"precision: {precision:.4f}", file=f)
    print(f"recall: {recall:.4f}", file=f)
    print(f"fbeta: {fbeta:.4f}",file=f)


# save metrics on slices to txt file
for feature in cat_features:
    slice(feature, test[cat_features], y_test , preds)




