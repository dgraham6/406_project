# U.S. Presidential Elections and S&P 500 Market Behavior

This repository contains exploratory and modeling work examining whether U.S. presidential
elections are associated with changes in S&P 500 market behavior, with a focus on volatility.

## Contents
- **EDA Notebook**: Exploratory data analysis of S&P 500
- **Stochastic Process Notebook**: Implementation of a data-driven stochastic return model and
  Monte Carlo simulations used to construct realistic null distributions for event-window
  volatility.

# Monte Carlo Hypothesis Testing for S&P 500 Market Analysis

Monte Carlo hypothesis testing is essential in this project because the volatility dynamics of financial markets violate the assumptions required for classical analytical tests. Daily S&P 500 returns exhibit time-varying volatility, heavy tails, and serial dependence, meaning we cannot rely on simple theoretical distributions (e.g., normality) to evaluate whether volatility during political events is unusually high. Instead, we build a data-driven null model that reconstructs typical market behavior and then simulate thousands of alternate return histories in which political events have no effect. This allows us to empirically approximate the sampling distribution of our event-window volatility statistic under realistic market conditions. By comparing the observed volatility to this Monte Carlo null distribution, we can rigorously test whether political events produce abnormal market responses. Thus, Monte Carlo testing provides a principled and flexible framework for inference in settings—like financial data—where analytic theory is inadequate.



# Monte_Carlo.ipynb

This notebook implements the full Monte Carlo hypothesis-testing framework used in our analysis of political-event volatility in the S&P 500. The notebook reconstructs the null data-generating process by estimating drift, extracting empirical standardized residuals, and preserving the observed time-varying volatility structure. It then simulates thousands of synthetic return paths under the null hypothesis that political events have no effect on market behavior. For each event date, the notebook computes the observed event-window volatility statistic and compares it to its simulated null distribution to generate Monte Carlo p-values. These simulations allow us to assess whether the volatility observed around real political events is unusually large relative to what would occur in typical market conditions. In short, the notebook provides the computational engine that powers the hypothesis test and produces the figures and results reported in the paper.



# Cross Validation for S&P 500 Market Analysis #

All of the data is located in the data folder of this branch. This branch holds the contributions made by Dervin Tian to the final project. 
To replicate the exact results found in the paper, go into the dervin folder where you will find all of the files used to replicate the results found in the paper.
The files to be taken note of within the dervin folder are the process_data.py script, where the predictions were created, and the hypo-test.py file where the 
hypothesis tests were conducted.

## process_data.py ##

In this script, the aim was to create the linear models for both the standard linear models as well as the election linear models. The data was also processed here in which 
we created a pandas dataframe that stored the S&P 500 data. The main features of this dataset were:
- Opening Price
- Closing Price
- High Price
- Low Price
- Volume

An additional column containing log returns was computed during preprocessing; however, this variable was intentionally excluded from the feature set. Because returns are 
directly derived from price levels, including them as predictors would introduce redundant information. Instead, we restrict the model to using only raw price variables, allowing 
it to learn return dynamics solely from price movements rather than from transformations of the target itself.

Under the current setup, executing process_data.py reproduces the exact results reported in the paper. To estimate the standard deviations required for the hypothesis tests, 
however, multiple simulations were conducted by varying the partitioning of the training and testing datasets across iterations. In the final iteration, the first 80% of the 
dataset was used for training and the remaining 20% for testing, which was the partition used for the results found in the paper. In earlier iterations, these splits differed, 
resulting in variation in the model outcomes and, consequently, a nonzero standard deviation for the test statistics.

## hypo-test.py ##

The purpose of this script is to set up a hypothesis test that can tell us whether or not the difference in the errors is statistically significant. A simple one-sided hypothesis
test was conducted with an implicit alpha level of 0.05 present. Running the tests on absolute difference between both the MAE and the RMSE produced large p-values that suggest that
the difference between the errors are not significant.

## Execution ##
To successfully reproduce the results found in the study, run the process_data.py to generate the predictions. The current setup for the process_data.py will split the dataset up 
with 80% training data and 20% testing data with the final 20% of dataset representing the testing dataset. The process_data.py will generate the prediction plot showing the 
prediction plotted against the actual returns. The script will also generate the MAE and RMSE for each linear model. There is a variable called models which holds the different
linear models, with one representing the standard linear model and another representing the election linear model.
