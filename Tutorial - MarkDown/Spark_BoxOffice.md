
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

# CIS5560 Term Project Tutorial


------
#### Authors: Tianyi Deng, Biran Li, Jiyang Liu, Yu-Lin Shih

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/19/2017



# LR Part:


# Prepare the Data
First, import the libraries you will need and prepare the training and test data


```python
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

```

# Data description
Budget: the budget of the movie, domesticgross: the revenue of movie in the United States, globalgross: the revenue of movie globally, duration: the length of the movie, language: the language of the movie, country: country that release the movie, imdb_score: the rating of movie, majorgenres: the major type of the movie


```python
sqlContext = SQLContext(sc)
movieschema = StructType([\
        StructField('Budget', DoubleType(), False),\
        StructField('domesticgross', DoubleType(), False),\
        StructField('globalgross', DoubleType(), False),\
        StructField('duration', DoubleType(), False),\
        StructField('language', IntegerType(), False),\
        StructField('country', IntegerType(), False),\
        StructField('imdb_score', DoubleType(), False),\
        StructField('majorgenres', IntegerType(), False),\
    ])
movieschema
```

# Insert SparkSession DataFrame from the file
click the insert to code button on the csv file then click on the insert SparkSession DataFrame


```python
from pyspark.sql import SparkSession

# @hidden_cell
# This function is used to setup the access of Spark to your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def set_hadoop_config_with_credentials_ce6b74a554044ac18da2684a94f2bf47(name):
    """This function sets the Hadoop configuration so it is possible to
    access data from Bluemix Object Storage using Spark"""

    prefix = 'fs.swift.service.' + name
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + '.auth.url', 'https://identity.open.softlayer.com'+'/v3/auth/tokens')
    hconf.set(prefix + '.auth.endpoint.prefix', 'endpoints')
    hconf.set(prefix + '.tenant', '571b641469df40cdb9cdf21a9ec8bdf6')
    hconf.set(prefix + '.username', 'bd693708437147129b446bf93514b247')
    hconf.set(prefix + '.password', 'k9a5-oNA.w#LaAW(')
    hconf.setInt(prefix + '.http.port', 8080)
    hconf.set(prefix + '.region', 'dallas')
    hconf.setBoolean(prefix + '.public', False)

# you can choose any name
name = 'keystone'
set_hadoop_config_with_credentials_ce6b74a554044ac18da2684a94f2bf47(name)

spark = SparkSession.builder.getOrCreate()

df_data_1 = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .load('swift://sparkProject.' + name + '/movieprojectazure.csv')
df_data_1.take(5)
```


```python
csv = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .schema(movieschema)\
  .load('swift://sparkProject.' + name + '/movieprojectazure.csv')
csv.show(5)
```

# Split data
select columns and Split data to 70% of training and 30% of testing


```python
# Select features and label
data = csv.select("Budget", "domesticgross", "duration",'language','country',"imdb_score",'majorgenres', col("globalgross").alias("label"))

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
```

# Define the Pipeline
Now define a pipeline that creates a feature vector and trains a regression model


```python
# Define the pipeline
assembler = VectorAssembler(inputCols = ["Budget", "domesticgross", "duration",'language','country',"imdb_score",'majorgenres'], outputCol="features")
lr = LinearRegression(labelCol="label",featuresCol="features")

pipeline = Pipeline(stages=[assembler, lr])
```

# Tune Parameters
You can tune parameters to find the best model for your data. To do this you can use the CrossValidator class to evaluate each combination of parameters defined in a ParameterGrid against multiple folds of the data split into training and validation datasets, in order to find the best performing parameters. Note that this can take a long time to run because every parameter combination is tried multiple times.


```python
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.01]).addGrid(lr.maxIter, [10, 5]).build()
# TODO: K = 2, you may test it with 5, 10
# K=2, 5, 10: Root Mean Square Error (RMSE): 13.2
cv = CrossValidator(estimator=pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, numFolds=10)

model = cv.fit(train)
```

# Test the Model
Now you're ready to apply the model to the test data.


```python
prediction = model.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show()
```

# Examine the Predicted and Actual Values
You can plot the predicted values against the actual values to see how accurately the model has predicted. In a perfect model, the resulting scatter plot should form a perfect diagonal line with each predicted value being identical to the actual value - in practice, some variance is to be expected. Run the cells below to create a temporary table from the predicted DataFrame and then retrieve the predicted and actual label values using SQL. You can then display the results as a scatter plot, specifying - as the function to show the unaggregated values.


```python
predicted.createOrReplaceTempView("regressionPredictions")
```


```python
%matplotlib inline
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

# Reference: http://standarderror.github.io/notes/Plotting-with-PySpark/
dataPred = spark.sql("SELECT trueLabel, prediction FROM regressionPredictions")
# convert to pandas and plot
regressionPredictionsPanda = dataPred.toPandas()
stuff = scatter_matrix(regressionPredictionsPanda, alpha=0.7, figsize=(6, 6), diagonal='kde')
```

# Retrieve the Root Mean Square Error (RMSE)
There are a number of metrics used to measure the variance between predicted and actual values. Of these, the root mean square error (RMSE) is a commonly used value that is measured in the same units as the prediced and actual values - so in this case, the RMSE indicates the average number of minutes between predicted and actual flight delay values. You can use the RegressionEvaluator class to retrieve the RMSE.


```python
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print "Root Mean Square Error (RMSE):", rmse
```

# GBT Part: 


# Import Spark SQL and Spark ML Libraries
First, import the libraries you will need:


```python
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator


from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer

```

# Load Source Data
The data for this exercise is provided as a CSV file containing details of movies. The data includes specific characteristics (or features) for each movie
You will load this data into a DataFrame and display it.


```python
df = spark.sql('select * from movieproject')

df.show(5)
```

# Data description
Budget: the budget of the movie, domesticgross: the revenue of movie in the United States, globalgross: the revenue of movie globally, duration: the length of the movie, language: the language of the movie, country: country that release the movie, imdb_score: the rating of movie, majorgenres: the major type of the movie


```python
# Calling cache on the DataFrame will make sure we persist it in memory the first time it is used.
df.cache()
```

# Visualize our data
Calling display() on a DataFrame in Databricks and clicking the plot icon below the table will let you draw and pivot various plot. Choose display type: Scatter plot. Drag columns under Values.



```python
#We can call display() on a DataFrame in Databricks to see a sample of the data.
display(df)

print "Our dataset has %d rows." % df.count()

#print the schema of our dataset to see the type of each column.
df.printSchema()
```

# Preprocess data
The DataFrame is currently using strings, so let's cast them.


```python
# The following call takes all columns (df.columns) and casts them using Spark SQL to a numeric type (DoubleType).
from pyspark.sql.functions import col  

# for indicating a column using a string in the line below
df = df.select([col(c).cast("double").alias(c) for c in df.columns])
df.printSchema()
```

# Split data into training and test sets
Our final data preparation step will split our dataset into separate training and test sets. We can train and tune our model as much as we like on the training set, as long as we do not look at the test set. After we have a good model (based on the training set), we can validate it on the held-out test set in order to know with high confidence our well our model will make predictions on future (unseen) data.


```python
# Split the dataset randomly into 70% for training and 30% for testing.
train, test = df.randomSplit([0.7, 0.3])
print "We have %d training examples and %d test examples." % (train.count(), test.count())
```


```python
display(train.select("budget", "globalgross"))
```

# Train a Machine Learning Pipeline
Now that we have understood our data and prepared it as a DataFrame with numeric values, let's learn an ML model to predict movie box(globalgross) in the future. Most ML algorithms expect to predict a single "label" column ("globalgross" for our dataset) using a single "features" column of feature vectors. For each row in our data, the feature vector should describe what we know: budget, country, duration etc., and the label should be what we want to predict (globalgross). We will put together a simple Pipeline with the following stages:
"VectorAssembler": Assemble the feature columns into a feature vector. "VectorIndexer": Identify columns which should be treated as categorical. This is done heuristically, identifying any column with a small number of distinct values as being categorical. "GBTRegressor": This will use the Gradient-Boosted Trees (GBT) algorithm to learn how to predict rental counts from the feature vectors. "CrossValidator": The GBT algorithm has several hyperparameters, and tuning them to our data can improve accuracy. We will do this tuning using Spark's Cross Validation framework, which automatically tests a grid of hyperparameters and chooses the best.

First, we define the feature processing stages of the Pipeline:
Assemble feature columns into a feature vector.
Identify categorical features, and index them.


```python
from pyspark.ml.feature import VectorAssembler, VectorIndexer
featuresCols = df.columns
featuresCols.remove('globalgross')
# This concatenates all feature columns into a single feature vector in a new column "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
# This identifies categorical features and indexes them.
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)
```

Second, we define the model training stage of the Pipeline. GBTRegressor takes feature vectors and labels as input and learns to predict labels of new examples.


```python
from pyspark.ml.regression import GBTRegressor
# gbt
gbt = GBTRegressor(labelCol="globalgross")

```

Third, we wrap the model training stage within a CrossValidator stage. CrossValidator knows how to call the GBT algorithm with different hyperparameter settings. It will train multiple models and choose the best one, based on minimizing some metric. In this example, our metric is Root Mean Squared Error (RMSE).


```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
# gbt
# Define a grid of hyperparameters to test:
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()
# We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
# Declare the CrossValidator, which runs model tuning for us.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)
```

Finally, we can tie our feature processing and model training stages together into a single Pipeline.


```python
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])

```

# Train the Pipeline!


```python
pipelineModel = pipeline.fit(train)
```

# Make predictions, and evaluate results
Calling transform() on a new dataset passes that data through feature processing and uses the fitted model to make predictions. We get back a DataFrame with a new column predictions


```python
predictions = pipelineModel.transform(test)
```


```python
It is easier to view the results when we limit the columns displayed to:
"globalgross": the true count of bike rentals
"prediction": our predicted count of bike rentals
```


```python
display(predictions.select("globalgross", "prediction", *featuresCols))

```


```python
rmse = evaluator.evaluate(predictions)
print "RMSE on our test set: %g" % rmse
```

# Visualization:
Plotting predictions vs. features(globalgross)


```python
display(predictions.select("globalgross", "prediction"))
```

# References:

Microsoft DAT202.3x Implementing Predictive Analytics with Spark in Azure HDInsight
Microsoft's DAT203x, Data Science and Machine Learning Essentials
Gautam, Geetika, and Divakar Yadav. Sentiment Analysis of Twitter Data Using Machine Learning Approaches and Semantic Analysis - IEEE Xplore Document. N.p., n.d. Web. 02 May 2017. 
"MovieLens." GroupLens. N.p., 18 Oct. 2016. Web. 02 May 2017. 
Concetta A., and Constanc H. McLaren. "Movie Data." Journal of Statistics Education, V17n1. N.p., 2009. Web. 02 May 2017. 
Chuansun7. "IMDB 5000 Movie Dataset." IMDB 5000 Movie Dataset | Kaggle. N.p., n.d. Web. 02 May 2017.
"The Academy Awards, 1927-2015." Academy of Motion Picture Arts and Sciences, n.d. Web. 02 May 2017. 



```python

```
