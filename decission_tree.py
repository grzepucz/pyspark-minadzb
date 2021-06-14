from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.stat import Correlation, ChiSquareTest
from plots import plot_corr_matrix, draw_table, plot_confusion_matrix
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from spark import data_columns_classification, data_columns_regression


def decision_tree_classification(data):
    train, test = data.randomSplit([0.7, 0.3])

    assembler = VectorAssembler(
        inputCols=data_columns_classification, outputCol='features')
    trained_vector_data = assembler.transform(train)
    tested_vector_data = assembler.transform(test)
    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel", handleInvalid="skip").fit(
        trained_vector_data)
    feature_indexer = \
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", handleInvalid="skip").fit(trained_vector_data)

    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, dt])
    model = pipeline.fit(trained_vector_data)
    predictions = model.transform(tested_vector_data)
    predictions.select("prediction", "indexedLabel", "features").show(5)

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    evaluator_score = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")
    accuracy = evaluator_acc.evaluate(predictions)
    score = evaluator_score.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)

    print("Test Error = %g " % (1.0 - accuracy))
    print("accuracy = ", accuracy)
    print("f1 score = ", score)
    print("weighted precision = ", precision)
    print("weighted recall = ", recall)

    print('DT CORRELATION')
    r1 = Correlation.corr(trained_vector_data, "features").head()
    print("DT Pearson correlation matrix:\n" + str(r1[0]))
    plot_corr_matrix(r1[0].toArray().tolist(), data_columns_classification, 'DT Pearson correlation matrix')

    r2 = Correlation.corr(trained_vector_data, "features", "spearman").head()
    print("DT Spearman correlation matrix:\n" + str(r2[0]))
    plot_corr_matrix(r2[0].toArray().tolist(), data_columns_classification, 'DT Spearman correlation matrix')

    print('DT HYPOTHESIS TESTING')
    r = ChiSquareTest.test(trained_vector_data, "features", "label").head()
    print("pValues: " + str(r.pValues))
    # draw_table(data_columns_classification, r.pValues, 'DT Chi pValues')
    print("degreesOfFreedom: " + str(r.degreesOfFreedom))
    # draw_table(data_columns_classification, r.degreesOfFreedom, 'DT Chi degreesOfFreedom')
    print("statistics: " + str(r.statistics))
    # draw_table(data_columns_classification, r.statistics, 'DT Chi statistics')

    metrics = MulticlassMetrics(
        predictions.select("prediction", "label").rdd.map(lambda lp: (float(lp.prediction), float(lp.label))))
    plot_confusion_matrix(metrics.confusionMatrix().toArray(), 'DT confusion matrix')


def decision_tree_regression(dataset):
    train, test = dataset.randomSplit([0.7, 0.3])

    assembler = VectorAssembler(
        inputCols=data_columns_regression, outputCol='features')
    trained_vector_data = assembler.transform(train)
    tested_vector_data = assembler.transform(test)
    feature_indexer = \
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", handleInvalid="skip").fit(trained_vector_data)
    dt = DecisionTreeRegressor(featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[feature_indexer, dt])
    model = pipeline.fit(trained_vector_data)
    predictions = model.transform(tested_vector_data)

    rmse_evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    mse_evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="mse")
    r2_evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2")
    mae_evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="mae")
    var_evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="var")
    mse = mse_evaluator.evaluate(predictions)
    rmse = rmse_evaluator.evaluate(predictions)
    r2 = r2_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    var = var_evaluator.evaluate(predictions)
    print("Mean Squared Error (MSE) on test data = %g" % mse)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    print("Mean Absolute Error (MAE) on test data = %g" % mae)
    print("Coefficient of Determination (R2) on test data = %g" % r2)
    print("Explained Variance on test data = %g" % var)
