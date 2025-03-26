# CapstoneUCBerkeleyAIML2025
Capstone UCBerkeley AIML March Madness Mania 2025
Capstone Project 2025 UC Berkeley AI\ML Course:

Project Title: March Machine Learning Mania 2025 Kaggle Competition (link)

Git Hub Link: 

Problem Statement\Business Understanding:

The goal of the contest is to predict the probability of team X beating team Y in NCAA basketball games based on previous history of the team performance. 

Data Understanding\Data Preparation:

The main parameters that influence the team performance is the team seed and the different scoring parameters within each game. The following parameters were taken into account, Num OT- Number of over time, WFGM - field goals made (by the winning team, WFGA - field goals attempted (by the winning team), WFGM3 - three pointers made (by the winning team), WFGA3 - three pointers attempted (by the winning team), WFTM - free throws made (by the winning team), WFTA - free throws attempted (by the winning team), WOR - offensive rebounds (pulled by the winning team), WDR - defensive rebounds (pulled by the winning team), WAst - assists (by the winning team), WTO - turnovers committed (by the winning team), WStl - steals (accomplished by the winning team), WBlk - blocks (accomplished by the winning team), WPF - personal fouls committed (by the winning team). 

Figure 1 and Figure 2 shows the dependency of seed difference on the win probability. Figure 3 shows feature engineering of the over time, taking into taking into account sum, medium, mean, max, min, std dev, skew and nunique. In the actual code, this was implemented for all the features. 






Figure 1: Win Probability by Seed Difference 




Figure 2: Win Rate by Seed %






Figure 3: Feature Engineering of NumOT, taking into account sum, medium, mean, max, min, std dev, skew and nunique


Data Preparation:

Data is checked for missing data and if any missed data, it was substituted with either 0 or -1 or imputed with mean depending on the variable. 




Modeling\Evaluation :

The following models were built and tested. Hyper parameter tuning and cross validation was performed to obtain the optimum model parameters. 

Table 1: Different Model build and evaluation of the model
#
Model
Log Loss
Brier Score
1
Dummy Classifier
3.5519
0.51309
2
Logistic Regression: Single Feature
0.55
0.19
3
Logistic Regression: All Features; Reduced dataset
0.57
0.19
4
Logistic Regression: All Features; Complete dataset
0.5
0.17
5
XGBoost Regression
0.001679
5.8614e-6
6
XGBoost Classifier
0.005675
0.0001139
7
Artificial Neural Network 
0.419
0.138


Logistic Regression:

A simple model using only the seed difference between the two teams was used first. With logistic regression, even with the single feature the Brier score improved from 0.513 to 0.19. 
Figure 4 represents the results from the Logistic regression model with single feature. Accuracy of 0.72 was obtained with this model. With feature engineering and increase dataset, the best accuracy that was achieved with logistic regression was 0.75. Results are shown in Figure 5. Ensemble models were explored to improve the accuracy and improve the Brier score. 


Figure 4: Win Probability by Seed Difference directly from data  and using Logistic Regression with seed difference as the input feature






Accuracy
0.75
Precision
0.75
Recall
0.75
F1-score
0.75


Figure 5: Confusion Matrix; Accuracy, Precision, Recall, F1-score and ROC Curve



XGBoost Regression and XGBoost Classifier:

XGBoost is an ensemble technique that is particularly used when accuracy is a key metric. Both XGBoost based regression and classifier were implemented. Hyperparameter search was performed to identity the optimum parameters. N_estimator of 5000 with a slow learning rate of 0.03 was chosen as the final optimum parameter. The results from the model were cross validated and the error metrics including log loss and Brier Score were estimated. The results are tabulated in Table 1. 

Artificial Neural Network

 ANN was implemented using  2 layers. Hyper parameter tuning was performed to identify the model parameters. The following parameters were identified as the best parameters. With this model the accuracy between the training and validation was not optimal as see in Figure 6




Figure 6: ANN Model #1 with 2 layer; Training Vs Validation Accuracy




Figure 7: ANN Model #2 with 3 layers,  Training Vs Validation Accuracy




Summary:

All the models faired better than the dummy classifier.
Logistic Regression showed best Brier score of 0.19
XGBoost Regression showed best Brier score of 5.86e-6
ANN showed best Brier score of 0.138
XGBoost was the best model for this problem


Next Steps:

Explore more advanced ANN models with multiple layers to improve the training and validation accuracy.



