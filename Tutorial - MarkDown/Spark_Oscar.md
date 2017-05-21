
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

# CIS5560 Term Project Tutorial


------
#### Authors: Tianyi Deng; Biran Li;Jiyang Liu;Yu-lin Shih

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/19/2017



# Analysis Oscar Nominations using DecisionTreeClassifier

In this tutorial, you will implement a classification model using DecisionTreeClassifier that uses features of a movie to predict whether or not the movie will be nominated as Best Picture in Oscar award.
You should follow the steps below to build, train and test the model :
1. Import Spark SQL and Spark ML Libraries
2. Load Source Data
3. Prepare Data
4. Split Data
5. Define the Pipeline
6. Run the Pipeline as an Estimator
7. Test the Pipeline Model
8. Train the model
9. Review Rate
10. Review the Error rate
11. Review the Area Under ROC

# Import Spark SQL and Spark ML Libraries


```python
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression 
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```

# Load Source Data
The data for this tutorial containing details of movie and provided as a CSV file. Select the columns that have most relative to won the Best Picture.
Load this data into a DataFrame and display it.


```python
"""movie_csv = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .schema(movieSchema) \
  .load('/FileStore/tables/lzdpk9mh1493750674219/movie_final_2.csv')
"""
movie_csv = spark.sql("SELECT ID, imdb_score, gross, director_facebook_likes, Award from movie_final_clean_2")

movie_csv.show(5)
```

# Prepare Data
Create a Boolean label field named label with the value 1 for movies that been nominate as Best Picture, or 0 that was not been nominated.


```python
data = movie_csv.select("ID", "imdb_score", "gross", "director_facebook_likes", ((col("Award") == "Best Picture").cast("Double").alias("label")))

data.show(5)
```

# Split Data
Split the data buy using 70% of the data for training, and reserve 30% for testing.


```python
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print "Training Rows:", train_rows, " Testing Rows:", test_rows
```

# Define the Pipeline
Define pipeline in several stages below.


```python
catVect = VectorAssembler(inputCols = ["ID"], outputCol="catFeatures")

catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures")

numVect = VectorAssembler(inputCols = ["imdb_score", "gross", "director_facebook_likes"], outputCol="numFeatures")

minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"], outputCol="features")

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

pipelineDt = Pipeline(stages=[catVect, catIdx, numVect, minMax, featVect, dt])

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.01)

pipelineLr = Pipeline(stages=[catVect, catIdx, numVect, minMax, featVect, lr])
```

# Run the Pipeline as an Estimator
Run the pipeline on the training data to train a model.


```python
piplineModel = []

piplineModel.insert(0, pipelineDt.fit(train))

piplineModel.insert(1, pipelineLr.fit(train))

print "Pipeline complete!"
```

# Test the Pipeline Model
Transform the test DataFrame using the pipeline to generate label predictions.


```python
prediction = [] 
predicted = []
for i in range(2):
  prediction.insert(i, piplineModel[i].transform(test))
  predicted.insert(i, prediction[i].select("features", "prediction", "probability", "trueLabel"))
  predicted[i].show(5)
```

# Review the Accuracy


```python
evaluator = []
for i in range(2):
  evaluator.insert(i, MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="accuracy"))
  accuracy = evaluator[i].evaluate(prediction[i])
  print("accuracy", i , " = ", accuracy)
```

# Reference
1. Microsoft DAT202.3x Implementing Predictive Analytics with Spark in Azure HDInsight
2. Microsoft's DAT203x, Data Science and Machine Learning Essentials
3. Gautam, Geetika, and Divakar Yadav. Sentiment Analysis of Twitter Data Using Machine Learning Approaches and Semantic Analysis - IEEE Xplore Document. N.p., n.d. Web. 02 May 2017. 
4. "MovieLens." GroupLens. N.p., 18 Oct. 2016. Web. 02 May 2017. 
5. Concetta A., and Constanc H. McLaren. "Movie Data." Journal of Statistics Education, V17n1. N.p., 2009. Web. 02 May 2017. 
6. Chuansun7. "IMDB 5000 Movie Dataset." IMDB 5000 Movie Dataset | Kaggle. N.p., n.d. Web. 02 May 2017.
7. "The Academy Awards, 1927-2015." Academy of Motion Picture Arts and Sciences, n.d. Web. 02 May 2017. 



```python

```
