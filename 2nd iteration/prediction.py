from sklearn import metrics, cross_validation

def regression(models, X, Y):
    for name, model in models:
        predicted = cross_validation.cross_val_predict(model, X, Y, cv=10)
        print (name)
        print("Mean squared error: %0.2f" % (metrics.mean_squared_error(Y, predicted)))
        print("Mean absolute error: %0.2f" % (metrics.mean_absolute_error(Y, predicted)))
        print("R^2 coefficient: %0.2f" % (metrics.r2_score(Y, predicted)))