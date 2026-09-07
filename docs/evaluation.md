## Checkpoint 1 Evaluation Protocol

The baseline uses a stratified 80/20 train/test split with `random_state=42`.
Categorical variables are one-hot encoded and numeric variables are scaled in
a single scikit-learn pipeline fitted only on the training data. The target is
`default_flag`, and `application_id` is excluded from the feature matrix.

Primary metrics are ROC AUC and Gini, where:

`Gini = 2 * ROC AUC - 1`

Accuracy, precision, recall, F1, and the confusion matrix are also recorded.
The runnable baseline writes these results to
`data/processed/checkpoint_1_metrics.json`. Results should be regenerated
after changing the source data or feature policy.
