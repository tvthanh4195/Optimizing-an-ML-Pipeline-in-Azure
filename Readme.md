# Azure ML Pipeline Optimization
## Project Overview
This project, a component of the Udacity Azure ML Nanodegree, involves the construction and optimization of an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. The performance of this model is then evaluated against an Azure AutoML run.
## Project Summary
The project utilizes data from the direct marketing campaigns of a Portuguese banking institution, which are primarily phone call-based. The dataset comprises 20 features such as age, job, and marital status. The target column consists of two categories (Yes and No), indicating whether the client subscribed to the bank's term deposit or not.
The objective of the algorithms developed using the Python SDK (with Hyperdrive) and AutoML is to accurately predict the likelihood of a potential client subscribing to the bank's term deposit. This aids in resource allocation by targeting clients most likely to subscribe.
The best-performing model was identified using the AutoML run, which was a Voting Ensemble model with an accuracy of 91.78%. However, the Logistic classifier trained using Hyperdrive achieved an accuracy of 91.44%, closely matching the accuracy of the Voting Ensemble model.
## Pipeline Details: Scikit-learn and Hyperdrive
### Scikit-learn
A Logistic Regression model was initially created and trained using Scikit-learn in the train.py script. The steps executed in the python script include:
- Importing the banking dataset using Azure TabularDataset Factory
- Cleaning and transforming the data using a cleaning function
- Splitting the processed data into a training and testing set
- Training an initial Logistic Regression model using Scikit-learn, specifying the values of two hyperparameters, C and max_iter. C denotes the inverse of the regularization strength, while max_iter represents the maximum number of iterations for the model to converge. These two parameters were initially passed in the python script for later optimization using Hyperdrive.
- Saving the trained model
The model, with parameters C = 0.1 and max_iter = 100, achieved an accuracy of 91.43%.
### Hyperdrive
The initially trained model is then optimized using Hyperdrive. Hyperdrive facilitates automatic hyperparameter tuning, which is typically computationally expensive and manual. By employing Hyperdrive, this process can be automated and experiments can be run in parallel to efficiently optimize hyperparameters.
The steps involved in implementing Hyperdrive include:
- Configuring the Azure cloud resources
- Setting up the Hyperdrive
- Executing the Hyperdrive
- Retrieving the model with the parameters that resulted in the best model
In the Hyperdrive configuration, two particularly beneficial parameters are included: RandomParameterSampling and BanditPolicy.

**RandomParameterSampling** is a parameter sampler that randomly selects hyperparameter values from a broad range specified by the user for model training. This approach is superior to a grid sweep as it is less computationally expensive and time-consuming and can select parameters that yield high accuracy. The random sampler also supports early termination of low-performance runs, thus conserving computational resources. The parameters passed to the random sampler were:
- C: 0.01,0.1,10,100
- max_iter: 50,100,150,200

**BanditPolicy** is an early termination policy that ends runs prematurely if they are not achieving the same performance as the best model. This also contributes to improving computational efficiency and saving time as it automatically terminates models with poor performance.
The best model had parameters of C = 10 and max_iter = 50, and achieved an accuracy of 91.44%.

## Implementation of AutoML
The following steps were executed to implement AutoML:
- The banking dataset was imported using Azure TabularDataset Factory
- The data was cleaned and transformed using the cleaning function in train.py
- AutoML was configured and a run was submitted to identify the model with the best performance
- The best model was saved
The model that performed the best was a Voting Ensemble model, achieving an accuracy of 91.78%. The hyperparameters of the model were:
- max_iter = 100
- multi_class = ovr
- n_jobs = 1
- penalty = 12
- random_state = None
- solver = saga
- tol = 0.0001
- verbose = 0
- warm_start = False

Comparison of PipelinesWhen comparing both pipelines, AutoML appears to have the advantage due to:
- Fewer steps required to find the best model (Simpler architecture)
- Higher accuracy achieved
The primary advantage of AutoML over Hyperdrive is AutoML's ability to easily test different algorithms. We might assume that the chosen model was the best for this problem and attempt to optimize the hyperparameters using Hyperdrive. However, there might be a model we haven't tested that could perform better than the chosen model, which is what occurred in this project.

