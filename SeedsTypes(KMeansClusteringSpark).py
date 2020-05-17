# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Seeds').getOrCreate()



df = spark.read.csv('/FileStore/tables/seeds_dataset.csv',header=True,inferSchema=True)
df.show()



df.printSchema()



from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler



df.columns



assembler = VectorAssembler(inputCols=df.columns,
                           outputCol = 'features')
final_data = assembler.transform(df)



final_data.printSchema()



from pyspark.ml.feature import StandardScaler
sc = StandardScaler(inputCol ='features',
                   outputCol = 'scaledFeatures')



scalerModel = sc.fit(final_data)
final_data = scalerModel.transform(final_data)



final_data.head(1)



kmeans = KMeans(featuresCol='scaledFeatures', k=3)
model = kmeans.fit(final_data)



wssse = model.computeCost(final_data)
wssse



centers = model.clusterCenters()
centers



predictions = model.transform(final_data)
predictions.show()












