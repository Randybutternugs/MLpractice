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
############ CORE LEARNING ALGORITHMS ############
############## Linear regression EX ##############
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
# Sometimes you can overfit which is when the model knows a dataset TOO well because its seen it too often
# This leads the model to do horribly when predicting new entries because its too familiar with the training data

# We must create an input function that converts/encodes pandas dataframe data into usable tf.data.Dataset objects that our model can read
# This also allows us to feed the data in batches
# So in general an input function breaks data into epochs and batches to  be fed to our model

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# CREATING THE MODEL
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier
# Now its alive, and is a linear estimator, but a blank slate


# TRAINING THE MODEL
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears console output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model
print(result)


# Now that we've trained our model we need to make an input function so that we can give it data and it will make predictions for us
result = list(linear_est.predict(eval_input_fn))
print(result[0])

#look at the person its predicting on at index 3
print(dfeval.loc[3])
# This gives us the result of person at index 3 and their chance of survival
print(result[3]['probabilities'][1])

#this tells us if they actually survived or not
print(y_eval.loc[3])

#the model predicted a 53% chance of survival and the person in reality did survive



############################################################################
#################### CLASSIFICATION ########################################
############################################################################

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

# This specific dataset seperates flowers into 3 different classes of species.

# Setosa
# Versicolor
# Virginica

# --The information about each flower is the following.

# sepal length
# sepal width
# petal length
# petal width

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv") # Saves file as iris_training.csv, gets it from link
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

#trains and tests into two different dataframes
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

#note all numbers are in cm
train.head()

train_y = train.pop('Species')
test_y = test.pop('Species')
train.head() # the species column is now gone

train.shape
#we have 120 entries with 4 features (columns)

#Input function
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# We use the DNN (Deep neural network) model because we may not be able to find a linear coorespondance in our data
# Now we build the model
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)


# Now we Train the model
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
# We include a lambda to avoid creating an inner function previously# We include a lambda to avoid creating an inner function previously
# lambda allows for one line functions ex:
# x = lambda: print('hi')
# x()
# That works

#PREDICTION function
# This allows a user to type in Sepal length, width, petal len and width and it will spit out predicted class

def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid:
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))
#YEAHAHHAHAHAHAHAHAB FIRST AI BABEY IT PREDICTED VIRGINICA WITH 95.3% CONFIDENCE
