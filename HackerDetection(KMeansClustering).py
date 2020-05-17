# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Hackers').getOrCreate()



df = spark.read.csv('/FileStore/tables/hack_data.csv',header=True,inferSchema=True)
df.show()



df.printSchema()



df.columns



features = df.select('Session_Connection_Time',
 'Bytes Transferred',
 'Kali_Trace_Used',
 'Servers_Corrupted',
 'Pages_Corrupted',
 'WPM_Typing_Speed')



from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler



assembler = VectorAssembler(inputCols=['Session_Connection_Time',
 'Bytes Transferred',
 'Kali_Trace_Used',
 'Servers_Corrupted',
 'Pages_Corrupted',
 'WPM_Typing_Speed'], outputCol='features')
final_data = assembler.transform(df)



from pyspark.ml.feature import StandardScaler
sc = StandardScaler(inputCol ='features',
                   outputCol = 'scaledFeatures')



scalerModel = sc.fit(final_data)
final_data = scalerModel.transform(final_data)



kmeans = KMeans(featuresCol='scaledFeatures', k=2)
model = kmeans.fit(final_data)



wssse = model.computeCost(final_data)
wssse



centers = model.clusterCenters()
centers




