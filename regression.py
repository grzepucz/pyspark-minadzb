from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.stat import Correlation, ChiSquareTest
from plots import plot_corr_matrix, draw_table, plot_confusion_matrix
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from spark import data_columns_classification, data_columns_regression


def regression_classification(dataset):
    train, test = dataset.randomSplit([0.7, 0.3])

    assembler = VectorAssembler(
        inputCols=data_columns_classification, outputCol='features', handleInvalid='skip')
    trained_vector_data = assembler.transform(train)

    tested_vector_data = assembler.transform(test)

    lr = LogisticRegression(maxIter=10, regParam=0.05, elasticNetParam=0.3, featuresCol="features", labelCol="label")
    model = lr.fit(trained_vector_data)

    tested_df = model.transform(tested_vector_data)

    prediction_and_labels = tested_df.select("prediction", "label")
    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    evaluator_score = MulticlassClassificationEvaluator(metricName="f1")
    print('TESTED MODEL MEASURES')
    print("Training set accuracy = " + str(evaluator_acc.evaluate(prediction_and_labels)))
    print("Training set F1-score = " + str(evaluator_score.evaluate(prediction_and_labels)))

    metrics = model.summary
    accuracy = metrics.accuracy
    print('MODEL MEASURES')
    false_positive_rate = metrics.weightedFalsePositiveRate
    true_positive_rate = metrics.weightedTruePositiveRate
    f_measure = metrics.weightedFMeasure()
    precision = metrics.weightedPrecision
    recall = metrics.weightedRecall
    print("Accuracy: %s\nFalse Positive Rate: %s\nTrue Positive Rate: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, false_positive_rate, true_positive_rate, f_measure, precision, recall))

    print('REGRESSION CORRELATION')
    r1 = Correlation.corr(trained_vector_data, "features").head()
    print("REGRESSION Pearson correlation matrix:\n" + str(r1[0]))
    plot_corr_matrix(r1[0].toArray().tolist(), data_columns_classification, 'REGRESSION Pearson correlation matrix')

    r2 = Correlation.corr(trained_vector_data, "features", "spearman").head()
    print("REGRESSION Spearman correlation matrix:\n" + str(r2[0]))
    plot_corr_matrix(r2[0].toArray().tolist(), data_columns_classification, 'REGRESSION Spearman correlation matrix')

    print('REGRESSION HYPOTHESIS TESTING')
    r = ChiSquareTest.test(trained_vector_data, "features", "label").head()
    print("pValues: " + str(r.pValues))
    # draw_table(data_columns_classification, r.pValues, 'REGRESSION Chi pValues')
    print("degreesOfFreedom: " + str(r.degreesOfFreedom))
    # draw_table(data_columns_classification, r.degreesOfFreedom, 'REGRESSION Chi degreesOfFreedom')
    print("statistics: " + str(r.statistics))
    # draw_table(data_columns_classification, r.statistics, 'REGRESSION Chi statistics')

    metrics = MulticlassMetrics(
        tested_df.select("prediction", "label").rdd.map(lambda lp: (float(lp.prediction), float(lp.label))))
    plot_confusion_matrix(metrics.confusionMatrix().toArray(), 'Regression confusion matrix')


def linear_regression(dataset):
    train, test = dataset.randomSplit([0.7, 0.3])
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    assembler = VectorAssembler(
        inputCols=data_columns_regression, outputCol='features')
    trained_vector_data = assembler.transform(train)
    tested_vector_data = assembler.transform(test)
    lrModel = lr.fit(trained_vector_data)
    predictions = lrModel.transform(tested_vector_data)

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
