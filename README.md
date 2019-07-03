# PySparkMLRegression

This repository contains example code for creating ML models using PySpark.

These examples were initially created to demonstrate how to find the threshold at which diastolic and systolic blood pressure result in a diagnosis of hypertension. However, I felt it would be more useful to provide examples based on a publicly available dataset, so I replaced diastolic and systolic values mapping to hypertension with examples using per capita gdp and life expectancy. 

The dataset used here, gapminder_all_binary.csv, is based on a dataset used for the software carpentry lesson "Plotting and Programming in Python" (https://swcarpentry.github.io/python-novice-gapminder/). I made one alteration - I added a binary field, "Over 65", indicating whether the life expectency for a particular country in 2007 was over or under 65 (1 for yes, 0 for no). I use this field for examples where I train a random forest (among other tree-based regressions) to determine the threshold for when this field should map to 1 or 0 (because this is a contrived example where I specifically coded these values to a threshold of 65, we know in advance that the ML model should return not too far off that predetermined threshold). 

There are a three workbooks here. 

Pyspark_scikit_learn_RF_binary_regression.ipynb is based on an example written by Hunter Mills at UCSF that uses scikit-learn on a PySpark3 kernel to find the threshold for systolic and/or diastolic blood pressure readings coded as hypertension. This approach allows you to use PySpark in a clustered environment for creating dataframes from SQL queries against a very large database, taking full advantage of the distributed environment. This example then switches to pandas and scikit-learn to build, fit, run, and access results on a random forest regression.

The other two workbooks build the models using SparkML rather than scikit-learn. Pyspark_continuous_regression provides an example for predicting life expectancy based on GDP as a continuous variable (using linear regression and decision trees). PySpark_binary_regression uses a decision tree and a random forest to predict a binary value (whether the life expectency is above 65).
