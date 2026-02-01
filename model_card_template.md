# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Gradient Boosting Classifier built using scikit-learn. It uses 100 estimators with a maximum depth of 5 and a random state of 42 for reproducibility. The model was created as part of a machine learning deployment project using the Census Bureau dataset.

## Intended Use
This model is intended to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related features from census data. It is designed for educational and demonstration purposes as part of a machine learning pipeline deployment project.

## Training Data
The model was trained on the UCI Census Income dataset (also known as the "Adult" dataset). The dataset contains 32,561 rows and 15 columns, including demographic features such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country. The target variable is "salary," which is binary: ">50K" or "<=50K." The data was split into 80% training and 20% testing using a random state of 42. Categorical features were encoded using one-hot encoding, and the target label was binarized using a label binarizer.

## Evaluation Data
The evaluation data consists of 20% of the original Census dataset, held out during the train-test split. The same preprocessing steps (one-hot encoding for categorical features and label binarization for the target) were applied using the encoders fitted on the training data.

## Metrics
The model was evaluated using three classification metrics: Precision, Recall, and F1-score (F-beta with beta=1).

Overall performance on the test set:
- Precision: 0.7945
- Recall: 0.6645
- F1: 0.7237

Performance on data slices (by categorical features such as workclass, education, marital status, occupation, relationship, race, sex, and native country) was also computed and saved to slice_output.txt.

## Ethical Considerations
The Census dataset contains sensitive demographic features such as race, sex, and native country. The model's predictions could reflect historical biases present in the training data. Care should be taken when interpreting the model's outputs, as they may disproportionately affect certain demographic groups. This model should not be used to make decisions that impact individuals without careful consideration of fairness and potential disparate impact across different groups.

## Caveats and Recommendations
The model is trained on Census data from 1994, which may not reflect current demographic or economic conditions. The model's performance varies across data slices, indicating potential fairness concerns for some subgroups. For production use, further analysis of bias and fairness across protected groups would be recommended. Additionally, hyperparameter tuning and experimenting with different algorithms could improve model performance.
