from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate(y_test, prob, pred_labels, test_df, label_map, features, importances):

    test_df[['prob_H_or_D','prob_A']] = prob
    inv_label = {v:k for k,v in label_map.items()}
    test_df['pred_outcome'] = [inv_label[l] for l in pred_labels]
    def pred_winner(row):
        if row['pred_outcome'] == 'H_or_D':
            return row['home_team']
        elif row['pred_outcome'] == 'A':
            return row['away_team']
        else:
            return 'Draw'
    test_df['pred_winner'] = test_df.apply(pred_winner, axis=1)

    if 'target' in test_df.columns:
        print("logloss:", log_loss(y_test, prob))
        print("accuracy:", accuracy_score(y_test, pred_labels))
        print("Confusion matrix (rows true, cols pred):")
        print(confusion_matrix(y_test, pred_labels))
        print("Precision:", precision_score(y_test, pred_labels, average='macro'))
        print("Recall:", recall_score(y_test, pred_labels, average='macro'))
    #Save the results to a csv
    out_path = '../data/predictions_2025_matches.csv'
    test_df.to_csv(out_path, index=False)

    feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_importance.values, y=feat_importance.index)
    plt.title("Feature Importance")
    plt.show()