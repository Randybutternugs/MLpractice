import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
# %tensorflow_version 2.x FOR NOTEBOOKS ONLY

#Check tf version to ensure we use tf 2.0
print(tf.version)


#-=-=-=-=-=RESHAPING=-=-=-=-=-
t = tf.zeros([5,5,5,5])
print(t)
# IF I want to take all these values in this tensor and flatten them out I do the following
t = tf.reshape(t, [625])
print(t)
# OR we can make it 125, and have tensorflow figure out the other dimension using -1
t = tf.reshape(t, [125, -1])
print(t)
#tensorflow figures out it should be 5 so the shape is 125, 5

#-=-=-=-=-=END RESHAPING=-=-=-=-=-

##################################################
############ MOD 3 ###############################
############ CORE LEARNING ALGORITHMS ############
##################################################
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


#USING the Titanic dataset that has data on each passenger on the ship we will explore it
#LOAD THE DATASET
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # Training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # Testing data

#.pop is used to cut out the survive column and save it in the variables below
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain.head()

# Overall information about the whole dataset such as mean age etc etc
dftrain.describe()

#shape of the dataframe
dftrain.shape
# 627 rows, 9 cols

# Time to visualize data about the age of people on the ship
dftrain.age.hist(bins = 20)
# This tells us that most people on the ship were about 25 years old/ 20-30 ys old (tells us about possible bias)

# Now a horizontal bar graph displaying the number of women vs nmber of men
dftrain.sex.value_counts().plot(kind = 'barh')
# This shows us that men are the most abundant sex on the ship

# Now we use a horizonatal bargraph to show the number of people in each class
dftrain['class'].value_counts().plot(kind = 'barh')
# This shows that most people were in Third class... neat!

# Now we compare the sex of passengers to their survival rate
pd.concat([dftrain, y_train], axis = 1).groupby('sex').survived.mean().plot(kind = 'barh').set_xlabel('% survive')
# We see that males had a 20% survival rate whereas females had an almost 80% survival rate

##########################################################################################

# NOW THAT WEVE COMPLETED GENERAL RESEARCH/EXPLORATION WE CAN MOVE ON TO ACTUALLY MAKING A MODEL
# FIRST WE PASTE IN OUR DATA AGAIN

#LOAD THE DATASET
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # Training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # Testing data

#.pop is used to cut out the survive column and save it in the variables below
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# categorize the columns based on the data they contain

#Categorical is non numeric, like male OR female; First or Second Class
#We will need to make this data into integers so that the model can understand it easier so male = 1 and female = 0 and so on
#This process is called ---encoding---
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

#The point of separating and categorizing the columns is so we can make unique processes that will help encode the data for the model.

#Now we iterate through the Categorical columns, find unique values
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
# This for loop basically finds all unique values for the given column and tensorflow says "Here is the x-value/feature i'm looking at and it has a y-value/label of 'male' or 'female'"
# This helps the model understand that the string 'male' or 'female' or that the destinations 'New York' are actual y-values or encodings not just random string jibberish

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
#its easier for the model to understand numeric data
# It knows that its just looking for an entry with shape (1,0) and the key is 'fare' or 'age'

# ********** THE TRAINING PROCESS ********** #

# LOADING IN BATCHES / BATCHING
# Note:
# We load data in 'batches' of 32 (you can raise or lower this number depending on hardware)
# every epoch is the number of times the model will see the whole dataset
# the more times the model sees the entire dataset the better it will determine how to estimate it
# so if we have 10 epochs the model will have seen the data 10 times
