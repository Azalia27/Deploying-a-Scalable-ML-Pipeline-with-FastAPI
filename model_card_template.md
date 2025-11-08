# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a binary classification model trained on the U.S. Census Income dataset. It predicts whether a person earns more than \$50K per year based on demographic and employment-related features.

The model is built using a simple **RandomForestClassifier** from scikit-learn. One-hot encoding is used for categorical features, and a LabelBinarizer is used to encode the label.

## Intended Use
The model is intended for educational purposes to demonstrate how to build a deployable ML pipeline using FastAPI, including data processing, model training, evaluation, and inference.

It is **not** intended for real-world decision making or deployment in production systems.

## Training Data
The model was trained on the **census.csv** dataset, which contains 32,561 examples with 15 features each. The features include categorical and numerical attributes such as education level, occupation, marital status, race, and sex.

## Evaluation Data
The evaluation was performed on a test set obtained via a train-test split (default 80/20 split) from the same dataset.

## Metrics
The model was evaluated using **Precision**, **Recall**, and **F1 Score**.

Overall performance:
- Precision: **0.7419**
- Recall: **0.6384**
- F1: **0.6863**

Additionally, performance metrics were computed on slices of data based on each categorical feature, and results were saved to `slice_output.txt`.

## Ethical Considerations
This model was trained on census data, which may contain historical biases. As such, predictions could reflect societal or demographic biases present in the original data.

This model is not intended for deployment in sensitive domains like employment, housing, or credit.

## Caveats and Recommendations
- The model may not generalize well to data outside of the U.S. Census dataset.
- Only basic preprocessing and default model parameters were used — no hyperparameter tuning or optimization was performed.
- Users should not rely on this model for decision-making without further validation and fairness analysis.
