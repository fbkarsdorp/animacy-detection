for selector_name, selector in (
        ("chi2", chi2),
        ("f1-anova", f_classif),
        ("regression-anova", partial(f_regression, center=False))):
    for window in (1, 2, 3, 4, 5, 10):
        X_train_idx, X_test_idx, y_train_idx, y_test_idx = train_test_split(
            range(len(X)), range(len(X)), test_size=0.2, random_state=1)
        X_train, y_train = [X[i] for i in X_train_idx], [y[i] for i in y_train_idx]
        X_test, y_test = [X[i] for i in X_test_idx], [y[i] for i in y_test_idx]
        stacker = FeatureStacker(Windower(window), WordEmbeddings(model))
        X_train = stacker.fit_transform(X_train, y_train)
        X_test = stacker.transform(X_test, y_test)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        # initialize a classifier
        clf = SGDClassifier(shuffle=True)
        # experiment with a feature_selection filter
        anova_filter = SelectPercentile(selector)
        percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
        # construct the pipeline
        pipeline = Pipeline([('anova', anova_filter), ('clf', clf)])
        # these are the parameters we're gonna test for in the grid search
        parameters = {
            'clf__class_weight': (None, 'auto'),
            'clf__alpha': 10.0**-np.arange(1,7),
            'clf__n_iter': (20, 50, 100, 200, np.ceil(10**6. / X_train.shape[0])),
            'clf__penalty': ('l2', 'elasticnet'),
            'anova__percentile': percentiles}
        grid_search = GridSearchCV(
            pipeline, param_grid=parameters, n_jobs=12, scoring='f1', verbose=1)
        print "Performing grid search..."
        print "pipeline:", [name for name, _ in pipeline.steps]
        print "Window:", window
        print "Feauture Selection", selector_name
        print "parameters:"
        pprint(parameters)
        grid_search.fit(X_train, y_train)

        print "Best score: %0.3f" % grid_search.best_score_
        print "Best parameters set:"
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print "\t%s: %r" % (param_name, best_parameters[param_name])

        print
        preds = grid_search.predict(X_test)
        print "Classification report after grid search:"
        print classification_report(y_test, preds)
        print

        print "Classification report on nouns after grid search:"
        noun_preds = []
        i = 0
        for idx in X_test_idx:
            for j, w in enumerate(X[idx]):
                if w[3] in ('noun', 'name'):
                    noun_preds.append(i + j)
            i += len(X[idx])
        print classification_report(preds[noun_preds], y_test[noun_preds])

        print "Fitting a majority vote DummyClassifier"
        dummy_clf = DummyClassifier(strategy='constant', constant=1)
        dummy_clf.fit(X_train, y_train)
        preds = dummy_clf.predict(X_test)
        print "Classification report for Dummy Classifier:"
        print classification_report(y_test, preds)

        print 'Fitting `subject=animate` classifier:'
        preds = [1 if w[4].startswith('su') else 0 for i in X_test_idx for w in X[i]]
        print "Classification report for `subject=animate` classifier:"
        print classification_report(y_test, preds)

        print 'Fitting `subject/object=animate` classifier:'
        preds = [1 if w[4].startswith(('su', 'obj')) else 0 for i in X_test_idx for w in X[i]]
        print "Classification report for `subject=animate` classifier:"
        print classification_report(y_test, preds)
