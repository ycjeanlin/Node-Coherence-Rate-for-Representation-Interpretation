from sklearn.linear_model import LogisticRegression

def test(z, data, solver="lbfgs", multi_class="auto", *args, **kwargs):
    train_z = z[data.train_mask]
    train_y = data.y[data.train_mask]
    test_z = z[data.test_mask]
    test_y = data.y[data.test_mask]
    clf = LogisticRegression(
        solver=solver, 
        multi_class=multi_class, 
        *args,
        **kwargs
    )
    clf = clf.fit(train_z, train_y)
    y_score = clf.predict_proba(z)
    acc = clf.score(test_z, test_y)
    return y_score, acc