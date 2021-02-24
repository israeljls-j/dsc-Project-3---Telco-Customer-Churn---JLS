Business Understanding
'Churn' is a major cost to the telecom industry and is defined as customer disengagement from a vendor to move to a 'better' vendor. Most of the costs relate to the initial discounts given to a customer and the initial cost of setting up a customer.
There are many factors involved in why a customer leaves: cost, slow to no resolutions of problems etc. but the primary reason is a bad experience with the company: outages, bad customer service experience, non-returned calls and feeling that they don't matter to the company.
Reasons for leaving are noted below:
Customer Service - 70% - Quality - 30% Price - 24% Functionality - 13% Other 30%
82% of companies claim that retention is cheaper than acquisition and a 2% increase in retention could reduce costs by 10%.The average churn rate for the industry is 10% to 60%

Problem Statement
Churn at Telco is currently 26.%. Churn adds significant costs to the client because of the initial discounts given and the high cost of adding a customer

Object
Using the data provided by the client(Telco), the objective is to predict the customers who will become 'churn', meaning that they will disengage with the client and move to another vendor.

Decision Trees

A decision tree model is a tree-like model with leaves extending from features which represent decisions and their outcomes.  The model endeavors to predict the best decisions for business problems.

Decision trees are useful in our TelCo churn analysis because the client would like to predict whether a current customer will ‘churn’ or disengage with the company to find a ‘better’ solution to their problems.  The model analyzes current churn data and predicts who will stay and who will leave which is important to TelCo so that they can interact with these likely churn customers to prevent them from leaving.

Preprocessing:

The information provided by TelCo included 19 features and 7043 classes

We began this process by reviewing the data from various perspectives:
Missing data
Duplicate data
Graphical data
Digging in to understand the data

 Other analysis showed a non relevant  column which we deleted.   We also noted that 9 columns looked redundant.  Initially we will keep these 9 redundant columns.  We also noted 3 columns of continuous data and categorized them so that we could have a dataset with all columns categorized.

Next we reviewed the data for multicollinearity using a heatmap and it showed clearly the multicollinearity of the  9 columns noted above. We deleted 8 of the columns leaving the total feature count at 10 features. Our revised heatmap showed no significant multicollinearity.

We were ready for modelling.

Our first model was fairly generic just to give us a baseline to improve.  As seen in the table below, Accuracy was 78 but recall of the 1 variable was only 41 percent.  The confusion matrix shows those results with 202 true positive(tp) and 1166 true negative(tn).   You can see that this model relies heavily on the ‘No’ answer which may not be reliable for the imbalanced target.  Both gini impurity (21.43) and entropy(15.4)  are low. Entropy shows the amount of info stored when using the model.  A low entropy score uses less pc storage and shows that the features are not very distinguishable.
Gini impurity shows the amount of impurity in the tree leaves and a low score is 0 while a ‘bad’ score is 1.0.  So the features in the model are not very distinguishable and have a relatively low impurity(wrong answers).

Our next phase was to select the best features and we used feature importance to select those best features.  Our results showed the following:






Our best features from order of importance are tenure, InTotalCharges_1, MonthlyCharges_1, tenure_1, PaymentMethod_Mailed check, PaymentMethod_Electronic check, PaymentMethod_Credit card(automatic), PaperlessBilling_Yes, Contract_Two year, Contract_One year, StreamingMovies_yes, StreamingTV_Yes, TechSupport_Yes and DevicePrtection_yes(13 items)



Metric
Baseline
Model 2
Model 3
Model 4
Accuracy
78
76
73
75
Recall 0
92
89
55
69
Recall 1
41
42
91
81
Confusion matrix
1166 tn
106 fn
287 fp
202 tp
1138 tn
286 fn
134 fp
203 tp
696 tn
576 fn
115 fp
1157 tp
 884 tn
240 fn
399 fp
1032 tp
Gini impurity
21.43


21.4
21.4
Entropy
15.4
16.5
16.4
16.4
CV score @ 3






.76 w/diff 1%
CV score @ 9






.80 w/diff 1%












These 13 items were kept and the others deleted. This next model(2) showed not much improvement. 

We made another change next by performing SMOTE to deal with the imbalance of the target.   significantly increased the Recall 1 rate and boosted the true positives from 200 to 1032. 

 Model 4 changed the max_depth to  9 and the recall-1 score went down but   Accuracy: 82, Recall  0: 74, Recall 1: 83, tp 1044, cv score of 80.0.  
The entropy and gini scores did not change substantially from the baseline model
Overall we saw good improvements and the final model is useful although we have many leaves which could probably be pruned.

The root node is StreamingMovies branching to tenure_! and Contract_One year









The tree tells us:
 People who have StreamingMovies split ½ yes and ½ no.
People who have one  year contracts but not StreamingTv are more likely to churn
People who have two year contracts and StreamingTV
People who pay by check are more likely to churn
Senior Citizens are more likely to churn


KNN

KNN is a methodology that uses ‘nearest neighbors’ to predict answers to business challenges.  In this case, TelCo could benefit by using this algorithm as it is ideal for classification data and Telco has substantially all classification data.  The training data is used to calculate the distance of a particular feature/class to its nearest neighbor within the total features.  With that data we are able to predict new data based on these distances.

We rely on distances between items to help us predict churn so it is important that the data is normalized.  Our first action was to create a pipeline using MinMax scaler and KNN.  This provides a baseline for improvement.

We chose recall as our defining metric as it uses the training data and counts the total correct positive predictions out of all predictions which is important for churn, especially given the imbalance in the target data.  We also used ROC/AUC to gain an understanding of the confusion matrix.  We chose precision as our X axis due to the imbalance of the target. 


Metric
Baseline
Best
Accuracy
50
75
Recall - 0
82
59
Recall - 1
50
91
ROC/AUC
66
75







