# Neural Network Charity Analysis

Unsupervised machine learning of Cryptocurrency data

## Overview

Alphabet Soup is a philanthropic foundation dedicated to helping organizations that protect the environment and improve people's wellbeing. The project involves analyzing a data set of Alphabet Soup donations to over 34,000 organizations, using neural networks to create a binary classifier that is capable of predicting whether applicants will be successful if funded. The following steps are used for the analysis:

* Pre-processing data for the neural network model
* Compile, train, and evaluate the model
* Attempt to optimize the model

### Purpose

The aim of this analysis is to determine the impact of each donation from AlphabetSoup to organizations in order to vet potential recipients. Sometimes an organization will recieve the money and then disappear, so this will help to ensure that the foundation's funds are being used effectively. The model will hope to predict which organizations are worth donating to and which are too high risk.

## Resources

Tools and software: Python 3.7.9, Jupyter Notebook 6.1.4, Visual Studio Code 1.54.3. Tensorflow, sklearn, OneHotEncoder, and Pandas libraries. 

Data set: [charity_data.csv](https://github.com/jkenning/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)

Base model code: [AlphabetSoupCharity.ipynb](https://github.com/jkenning/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb)
Results: [trained_application.h5](https://github.com/jkenning/Neural_Network_Charity_Analysis/blob/main/trained_application.h5)

Attempted optimized model code: [AlphabetSoupCharity_Optimization.ipynb](https://github.com/jkenning/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb)
Results: [optimized_application.h5](https://github.com/jkenning/Neural_Network_Charity_Analysis/blob/main/optimized_application.h5)



## Summary of Results

### Data Pre-processing

* As the objective of the analysis is to predict whether a charitable donation is successful, the binary data column `IS_SUCCESSFUL` is therefore the target variable for the neural network.
* The other columns in the data set were used as features; with binning of the highly unique-valued `APPLICATION_TYPE` and `CLASSIFICATION` features, encoding of categorical variables, splitting of test and training data sets, and data scaling performed. 
* The only columns not used in the analysis were the non-beneficial ID columns `EIN` and `NAME`, which were dropped.

### Compiling, Training, and Evaluating the Model

The neural network was made using two hidden layers containing 80 and 30 neurons respectively. The input layer consists of 44 features and the first hidden layer has roughly twice that number of neurons. The number of hidden neurons is decreased in subsequent layers to get closer to pattern/feature extraction and identify the target class. The ReLU activation function is used for the hidden layers to help characterize non-linear relationships in the data. As we are looking for a binary outcome (successful or not), the output layer has a single neuron and uses the sigmoid activation function to transform to a range between 0 and 1 and classify the data. 

![](https://github.com/jkenning/Neural_Network_Charity_Analysis/blob/main/Images/model_summary.PNG)

Figure. 1 - Model Summary

The accuracy of the model was ~72-73% which is a little under the minimum target accuracy of 75%. As a result the model performance is unsatifactory for predicting funding success.

![](https://github.com/jkenning/Neural_Network_Charity_Analysis/blob/main/Images/model_evaluation.png)

Figure. 2 - Model Evaluation Results

In order to try and increase model performance the following steps were taken individually, and then in combination:

* The `INCOME_AMT` and `ASK_AMT` columns were dropped from the analysis to see if these variables were causing confusion in the model
* Increased (then decreased) the number of 'other' values in the `APPLICATION_TYPE` and `CLASSIFICATION` bins
* More neurons were added to the first and second hidden layers
* An additional third hidden layer was added
* Different activation function was used for the second and third hidden layers
* Doubled the number of epochs to the training regimen

![](https://github.com/jkenning/Neural_Network_Charity_Analysis/blob/main/Images/optimized_summary.png)

Figure. 4 - "Optimized" Model Summary

However, none of these particular tested modifications steps were able to increase model performance which remained within a few percentage points of the original base model. 

![](https://github.com/jkenning/Neural_Network_Charity_Analysis/blob/main/Images/optimized_evaluation.png)

Figure. 3 - "Optimized" Model Evaluation Results

## Summary

Neither the base model or the test steps implemented to improve model accuracy were able to reach the desired target threshold of 75% accuracy. This suggests that there may be unidentified relationships in the original data that are not being captured by the model and further investigations and pre-processing my be required prior to running the model. One way to solve this may be by adding more data entries and/or additional data fields. Neural networks generally require a lot of data to be effective and can be prone to over-fitting and generalization. Because of the "black box" nature of the neural network, we dont know why it determines a certain output and to what degree each independent variable is influencing the target variable.

Alternatively, a more suitable model for this problem might be a Randon Forest Classifier, as the use of multiple decision trees make it a strong predictive tool for a binary classification case such as this. They do not require normalization and are good with handling tabular data. In this case it would be worth testing how this model performs against the Neural Network using this data set. 