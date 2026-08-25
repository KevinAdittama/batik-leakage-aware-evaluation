| model_group | model | cv_f1_macro_mean | cv_f1_macro_std | external_f1_macro | external_balanced_accuracy | external_mcc | external_recall_batik | external_recall_non_batik |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen DL features | MobileNetV2 | 0.960 | 0.038 | 0.866 | 0.867 | 0.740 | 0.800 | 0.933 |
| Frozen DL features | ResNet18 | 0.919 | 0.031 | 0.833 | 0.833 | 0.668 | 0.867 | 0.800 |
| Classical ML | SVM (RBF) | 0.911 | 0.043 | 0.661 | 0.667 | 0.346 | 0.533 | 0.800 |
| Classical ML | Random Forest | 0.904 | 0.031 | 0.614 | 0.617 | 0.237 | 0.533 | 0.700 |
| Classical ML | Logistic Regression | 0.823 | 0.064 | 0.495 | 0.500 | 0.000 | 0.600 | 0.400 |
