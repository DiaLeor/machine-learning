<p>Taking notes on the <a href="https://www.edx.org/course/data-science-machine-learning">Data Science: Machine Learning</a> course; practicing what I've learned during the previous <a href="https://www.edx.org/course/data-science-productivity-tools">Data Science: Productivity Tools</a> course. For most effective navigation in RStudio, document outine (Ctrl+Shift+O) aligns with the following .R contents:<br></p>
<br>
<p>--- Directory: <a href="https://github.com/DiaLeor/machine-learning/tree/main/1_intro_to_ml">1_intro_to_ml</a> ---<br>
<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/1_intro_to_ml/1.1-notation.R">1.1-notation.R</a><br>
- Section 1.1 - Notation<br>
<br>
&emsp;&emsp;- Notation - a brief overview of machine learning notation<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Example: Zip Code Reader - machine learning notation continued<br>
&emsp;&emsp;<br></p>
<p><br>
<br></p>
<p>--- Directory: <a href="https://github.com/DiaLeor/machine-learning/tree/main/2_ml_basics">2_ml_basics</a> ---<br>
<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/2_ml_basics/2.1-evaluation.R">2.1-evaluation.R</a><br>
- Section 2.1 - Basics of Evaluation Machine Learning Algorithms<br>
<br>
&emsp;&emsp;- Evaluation Metrics - what we mean when we say one approach is "better"<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Confusion Matrix - understanding the confusion matrix<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Sensitivity, Specificity, and Prevalence - defining metrics for binary outcomes<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Balanced Accuracy and F1 Score - preferred one-number-summary metrics<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Prevalence Matters in Practice - when prevalence is close to either 0 or 1<br>
&emsp;&emsp;<br>
&emsp;&emsp;- ROC and Precision-Recall Curves - comparing methods graphically by plotting both<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Loss Funciton - a function which can be applied to both categorical and continuous data.
<br>
&emsp;&emsp;<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/2_ml_basics/2.2-conditional-probabilities.R">2.2-conditional-probabilities.R</a><br>
- Section 2.2 - Conditional Probabilities<br>
<br>
&emsp;&emsp;- Conditional Probabilites - using Baye's Rule to estimate conditional probabilities<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Conditional Expectations - the expected value has an attractive mathematical probability that minimizes the MSE<br>
&emsp;&emsp;<br></p>
<p><br>
<br></p>
<p>--- Directory: <a href="https://github.com/DiaLeor/machine-learning/tree/main/3_prediction">3_prediction</a> ---<br>
<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/3_prediction/3.1-linear-regression.R">3.1-linear-regression.R</a><br>
- Section 3.1 - Basics of Evaluation Machine Learning Algorithms<br>
<br>
&emsp;&emsp;- Linear Regression for Prediction - although it can be too rigid to be useful, it works rather well for some challenges<br>
&emsp;&emsp;<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/3_prediction/3.2-smoothting.R">3.2-smoothing.R</a><br>
- Section 2.2 - Smoothing<br>
<br>
&emsp;&emsp;- Smoothing - detecting trends in the presence of noisy data in cases in which the shape of the trend is unknown<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Bin Smoothing - grouping data points into strata in which the value of f(x) can be assumed to be constant in a window<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Local Weighted Regression (loess) - assuming that the function is locally linear, permitting us to consider larger window sizes<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Beware of Default Smoothing Parameters - by default: loess fits parabolas, not lines<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Connecting Smoothing to Machine Learning - smoothing approaches may provide an improvement in capturing the non-linear nature of a trend<br>
&emsp;&emsp;<br></p>
<p><br>
<br></p>
<p>--- Directory: <a href="https://github.com/DiaLeor/machine-learning/tree/main/4_cross_validation_kNN">4_cross_validation_kNN</a> ---<br>
<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/4_cross_validation_kNN/4.1-kNN.R">4.1-kNN.R</a><br>
- Section 4.1 - k-Nearest Neighbors<br>
<br>
&emsp;&emsp;- k-Nearest-Neighbors (kNN) - basics of kNN<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Overtraining and Oversmoothing (kNN) - what it means to overtrain and oversmooth in kNN<br>
&emsp;&emsp;<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/4_cross_validation_kNN/4.2-cross-validation.R">4.2-cross-validation.R</a><br>
- Section 4.2 - Cross Validation<br>
<br>
&emsp;&emsp;- Choosing k - how to choose an approprite k value<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Mathematical Description of Cross-Validation - <br>
&emsp;&emsp;<br>
&emsp;&emsp;- k-fold Cross-Validation - evaluating the performance of a predictive model by partitioning the dataset into subsets<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Bootstrap - details on the Bootstrap method<br>
&emsp;&emsp;<br></p>
<p><br>
<br></p>
<p>--- Directory: <a href="https://github.com/DiaLeor/machine-learning/tree/main/5_caret">5_caret</a> ---<br>
<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/5_caret/5.1-caret.R">5.1-caret.R</a><br>
- Section 5.1 - The Caret Package<br>
<br>
&emsp;&emsp;- Caret Package - introduction to the caret package<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Fitting with Loess - basics of fitting with loess<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Trees and Random Forests - introduction to trees and random forests<br>
&emsp;&emsp;<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/5_caret/5.2-titanic.R">5.2-titanic.R</a><br>
- Section 5.2 - Titanic Exercises<br>
<br>
&emsp;&emsp;- Titanic Exercises Pt. 1 - <br>
&emsp;&emsp;<br>
&emsp;&emsp;- Titanic Exercises Pt. 2  - []<br>
&emsp;&emsp;<br></p>
<p><br>
<br></p>
<p>--- Directory: <a href="https://github.com/DiaLeor/machine-learning/tree/main/6_model_fitting_recommendation">6_model_fitting_recommendation</a> ---<br>
<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/6_model_fitting_recommendation/6.1-MNIST.R">6.1-MNIST.R</a><br>
- Section 6.1 - Case Study: MNIST<br>
<br>
&emsp;&emsp;- MNIST Case Study: Preprocessing - []<br>
&emsp;&emsp;<br>
&emsp;&emsp;- MNIST Case Study: kNN - []<br>
&emsp;&emsp;<br>
&emsp;&emsp;- MNIST Case Study: Random Forest - []<br>
&emsp;&emsp;<br>
&emsp;&emsp;- MNIST Case Study: Variable Importance - []<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Ensembles - []<br>
&emsp;&emsp;<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/6_model_fitting_recommendation/6.2-recommendation.R">6.2-recommendation.R</a><br>
- Section 6.2 - Recommendation Systems<br>
<br>
&emsp;&emsp;- Recommendation Systems - []<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Building a Recommendation System - []<br>
&emsp;&emsp;<br>
-- <a href="https://github.com/DiaLeor/machine-learning/blob/main/6_model_fitting_recommendation/6.3-regularization.R">6.3-regularization</a><br>
- Section 6.1 - Regularization<br>
<br>
&emsp;&emsp;- Regularization - []<br>
&emsp;&emsp;<br>
&emsp;&emsp;- Matrix Factorization - []<br>
&emsp;&emsp;<br>
&emsp;&emsp;- SVD and PCA - []<br>
&emsp;&emsp;<br>
