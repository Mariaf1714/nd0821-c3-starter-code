from ml.model import compute_model_metrics


def slice(feature, df, y, preds):
    """ Function for computing model metrics on data slices of a feature."""
    df['label_value'] = y
    df['score'] = preds
    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]
        precision, recall, fbeta = compute_model_metrics(
            df_temp['label_value'], df_temp['score'])
        with open("slice_output.txt", "a") as f:
            print(f"Feature: {feature}", file=f)
            print(f"Class: {cls}", file=f)
            print(f"{cls} precision: {precision:.4f}", file=f)
            print(f"{cls} recall: {recall:.4f}", file=f)
            print(f"{cls} fbeta: {fbeta:.4f}", file=f)
            print("\n", file=f)
