# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


df=pd.DataFrame(...)

with pd.ExcelWriter('test.xls') as xlw:
    df.to_excel(xlw,'Sheet 1')
    df2.to_excel(xlw,'Sheet 2')

def processSubSet:
    X_train, X_test, y_train, y_test = train_test_split(X[list(feature_set)], y)
    model = LinearRegression()
    regr = model.fit(X_train, y_train)
    predicted = regr.predict(X_test)
    mean_obs = np.mean(y_test)
    ss_res = np.sum((y_test - predicted)**2)
    ss_tot = np.sum((y_test - mean_obs)**2)
    r = 1-(ss_res/ss_tot)
    return {"model":regr, "RSS":ss_res, "R_sqr":r}


X = np.array([[0], [1], [2], [3], [4], [5]])
labels = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
cv = KFold(len(labels), n_folds=3)
clf = SVC()
ypred_all = np.chararray((labels.shape))
i = 1
for train_index, test_index in cv.split(X):
    print("iteration", i, ":")
    print("train indices:", train_index)
    print("train data:", X[train_index])
    print("test indices:", test_index)
    print("test data:", X[test_index])
    clf.fit(X[train_index], labels[train_index])
    ypred = clf.predict(X[test_index])
    print("predicted labels for data of indices", test_index, "are:", ypred)
    ypred_all[test_index] = ypred
    print("merged predicted labels:", ypred_all)
    i = i+1
    print("=====================================")
y_cross_val_predict = cross_val_predict(clf, X, labels, cv=cv)
print("predicted labels by cross_val_predict:", y_cross_val_predict)
