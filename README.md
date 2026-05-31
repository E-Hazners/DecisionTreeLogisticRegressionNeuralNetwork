# Task from the Introduction to Intelligence Systems course
Practical work N1. Classification problem.

Part 1. Data vectorization and preparation for the problem of classification
Tasks:
1.1.Chose dataset. You can either choose your own data set or one of the proposed ones.
1.2. Load data set and transform it into the form of “numpy array”
1.3.Display information about the dimension of the array (the number of observations and
the number of variables).
1.4.Delete the rows with empties in data
1.5.Convert categorical variables from string format to binary (preparation for classification
task). Apply one-hot encoding.
1.6.Display the first 5 observations.
1.7.Divide the dataset into two subsamples in the proportion of 0.75 and 0.25. 75% of the
observations should be in the first subsample, and the remaining 25% in the second.

Part 2. Decision tree creation and performance evaluation
Tasks:
2.1.The selected and prepared dataset should be divided into subsamples for training and
validation. Cross-validation is allowed.
2.2.Implement/run an algorithm for building a decision tree (recommended "Iterative
Dichotomiser" algorithm)
2.3.Train the algorithm on the training set and evaluate the performance on the test set.
2.4.Display the number of leaves, depth (depth), metric value (gini by default)
2.5.Calculate and display the performance for the training and test subsamples.
2.6.Display the trained model graphically.

Part 3. Logistic regression
Tasks:
3.1.The selected and prepared dataset should be divided into subsamples for training and
validation. Cross-validation is allowed.
3.2.Implement/run an algorithm for logistic regression model creation
3.3.Train the algorithm on the training set and evaluate the performance on the test set.
3.4.Display the model’s parameters (coefficients and intercept)
3.5.Calculate and display the performance for the training and test subsamples.

Part 4. Application of a multilayer neural network with regularization for classification
problems
Tasks:
4.1.The selected and prepared dataset should be divided into subsamples for training and
validation. Cross-validation is allowed.
4.2.Implement/run an algorithm for multi-layer neural network training
4.3.Train the algorithm on the training set and evaluate the performance on the test set.
4.4.Display the model’s parameters of the model (vector of weights and intercept)
4.5.Calculate and display the performance for the training and test subsamples
4.6.To study how the variation of the epoch (max_iter) parameter affects the performance
of the model. Display graphically.
4.7. Learn how changing depth (number of hidden layers) and width (number of neurons per
layer) affects the model performance. Display graphically
