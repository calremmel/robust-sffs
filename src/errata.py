def permtest_evaluate(config_file="config.yml", features=None):
    """[summary]

    Args:
        config_file (str, optional): [description]. Defaults to "config.yml".
        k (str, optional): [description]. Defaults to "auto".
        features ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]

    Yields:
        [type]: [description]
    """
    config = load_config(config_file)
    k = self.best_k

    true = joblib.load(config["SELECTOR"])
    permuted = joblib.load(config["PERMUTATION_TEST"])
    # get true cv scores
    if type(features) == list:
        X, y = load_data(config["MAIN"], config_file=config_file)
        classifier = build_classifier(config_file=config_file)
        stratkfold = RepeatedStratifiedKFold(
            n_splits=config["CV"]["FOLDS"],
            n_repeats=config["CV"]["REPEATS"],
            random_state=88,
        ).split(X, y)
        cv = list(stratkfold)
        true_scores = cross_val_score(classifier, X[features], y, cv=cv)
    else:
        true_scores = list(true.steps[-1][1].get_metric_dict()[k]["cv_scores"])
        true_scores = [
            true_scores[i : i + config["CV"]["FOLDS"]]
            for i in range(0, len(true_scores), config["CV"]["FOLDS"])
        ]
        true_scores = [np.mean(x) for x in true_scores]
    # get cv scores from permutation test
    permuted_scores = []
    for i in permuted:
        scores = list(permuted[i].steps[-1][1].get_metric_dict()[k]["cv_scores"])
        permuted_scores.append(np.mean(scores))
    plot_filename = plot_permutation_violins(
        true_scores, permuted_scores, config_file=config_file
    )
    print("Permutation test plot saved at {}".format(plot_filename))
    pass


def output_predictions(config_file="config.yml", k="auto", features=None):
    """Outputs a .csv file of predicted probabilities and labels, annotated with metadata if available.

    Args:
        k ([type], optional): [description]. Defaults to config['K_FEATURES'].
        features ([type], optional): [description]. Defaults to None.

    Returns:
        report (DataFrame): DataFrame of predictions and probabilities.
    """
    config = load_config(config_file)
    k = self.best_k
    if type(features) != list:
        features = get_feature_names(k, config_file=config_file)
    if config["META_COLS"] != "N":
        X, y, meta = load_data(
            config["MAIN"], config_file=config_file, meta_return=True
        )
    else:
        X, y = load_data(config["MAIN"], config_file=config_file)

    classifier = build_classifier(config_file=config_file)
    classifier.fit(X[features], y)
    y_pred = classifier.predict(X[features])
    y_prob = classifier.predict_proba(X[features])[:, 1]
    group = ["MAIN" for x in y_pred]
    report = pd.DataFrame(
        {"group": group, "target": y, "prediction": y_pred, "probability": y_prob}
    )
    if config["META_COLS"] != "N":
        report = pd.concat([meta, report], axis=1)

    if config["TEST"]:
        if config["META_COLS"] != "N":
            Xt, yt, metat = load_data(
                config["TEST"], meta_return=True, config_file=config_file
            )
        else:
            Xt, yt = load_data(config["TEST"], config_file=config_file)
        yt_pred = classifier.predict(Xt[features])
        yt_prob = classifier.predict_proba(Xt[features])[:, 1]
        groupt = ["TEST" for x in yt_pred]
        report_test = pd.DataFrame(
            {
                "group": groupt,
                "target": yt,
                "prediction": yt_pred,
                "probability": yt_prob,
            }
        )
        if config["META_COLS"] != "N":
            report_test = pd.concat([metat, report_test], axis=1)
        report = pd.concat([report, report_test], axis=0)
    report.to_csv(config["REPORT"], index=False)
    print("Predictions saved at {}".format(config["REPORT"]))
    return report


######################
# PLOTTING FUNCTIONS #
######################


def plot_permutation_violins(
    pipeline_scores, permuted_scores, config_file="config.yml"
):
    """Create violin plots of true vs permuted pipeline runs.

    Args:
        pipeline_scores (list): List of accuracy scores, true labels.
        permuted_scores (list): List of accuracy scores, permuted labels.

    Returns:
        plot_filename (str): Output file location.
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "Accuracy vs Run | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_permutation_violins.png".format(
        now, config["RUN_NAME"]
    )

    pipeline = pd.DataFrame({"Run": "True", "Accuracy": pipeline_scores})
    permuted = pd.DataFrame({"Run": "Permuted", "Accuracy": permuted_scores})
    for_plotting = pd.concat([pipeline, permuted], axis=0)

    _, p = ttest_ind(pipeline_scores, permuted_scores)
    d, _ = cliffsDelta(pipeline_scores, permuted_scores)
    p = process_pval(p)
    d = str(round(d, 3))

    for_plotting["pval"] = p
    for_plotting["cliffsdelta"] = d
    for_plotting.to_csv(plot_filename.split(".")[0] + ".csv")
    current_palette = sns.color_palette()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        x="Run",
        y="Accuracy",
        data=for_plotting,
        bw="scott",
        cut=0,
        palette=[current_palette[1], current_palette[0]],
        ax=ax,
    )
    plt.axhline(0.5, color="r", linestyle="--")
    plt.axhline(np.mean(pipeline_scores), color="darkorange", linestyle="--")
    plt.axhline(np.mean(permuted_scores), color="darkblue", linestyle="--")
    plt.ylim([-0.1, 1.1])
    plt.title(plot_title)
    plt.text(
        0.5,
        0.1,
        "P Value: " + p + "\nCliff's Delta: " + d,
        horizontalalignment="center",
    )
    plt.tight_layout()
    plt.savefig(plot_filename)
    return plot_filename


def plot_feature_search(
    pipeline, X=None, y=None, X_test=None, y_test=None, config_file="config.yml"
):
    """Plots CV accuracy vs number of features. Optionally includes test set for comparison.

    Args:
        pipeline (sklearn pipeline): Feature selection pipeline.
        X (array, optional): Matrix of features. Defaults to None.
        y (array, optional): Array of labels. Defaults to None.
        X_test (array, optional): Matrix of features, test set. Defaults to None.
        y_test (array, optional): Array of labels, test set. Defaults to None.

    Returns:
        [type]: [description]
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)
    plot_title = "SFFS Accuracy vs Number of Features | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    search_object = pipeline.steps[-1][1]
    clf = search_object.estimator

    crossval_scores = pd.DataFrame(
        {
            key: val
            for key, val in zip(
                search_object.get_metric_dict().keys(),
                [
                    search_object.get_metric_dict()[x]["cv_scores"]
                    for x in search_object.get_metric_dict().keys()
                ],
            )
        }
    )
    crossval_scores = crossval_scores.melt()

    if type(X_test) == pd.core.frame.DataFrame:
        test_index = []
        test_accuracy = []
        evaluator = build_classifier()
        for x in search_object.get_metric_dict().keys():
            test_features = get_feature_names(x, config_file=config_file)
            print("Features for set {}:\n{}".format(x, test_features))
            evaluator.fit(X[test_features], y)
            predictions = evaluator.predict(X_test[test_features])
            accuracy = accuracy_score(y_test, predictions)
            test_index.append(x)
            test_accuracy.append(accuracy)

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.lineplot(
        x="variable",
        y="value",
        data=crossval_scores,
        ci="sd",
        ax=ax,
        label="Cross-Validation",
    )
    if type(X_test) == pd.core.frame.DataFrame:
        plt.plot(test_index, test_accuracy, label="Test Accuracy")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.title(plot_title)

    num_features = len(crossval_scores.variable.unique())
    plt.xticks(list(range(1, num_features + 1)), rotation=90)
    plt.ylim([-0.05, 1.15])
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plot_filename = "../output/{}_{}_sffs_plot.png".format(now, config["RUN_NAME"])
    plt.savefig(plot_filename)
    return plot_filename


def plot_coefficients(k="auto", config_file="config.yml"):
    """Plots coefficient bars. Logistic Regression only.

    Args:
        k (int, optional): Feature set size to plot. Defaults to config file setting.
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "Coefficients | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_coefficients.png".format(now, config["RUN_NAME"])

    features = get_feature_names(k, config_file=config_file)
    coefficients = get_coefficients(features, config_file=config_file)

    df = pd.DataFrame({"Feature": features, "Coefficient": coefficients})

    largest_coefficient = max(np.abs(coefficients))
    ymax = largest_coefficient + 0.10 * largest_coefficient

    df = df.sort_values("Coefficient", ascending=False)
    df["Color"] = df.Coefficient.apply(lambda x: "#7cfc00" if x >= 0 else "#551a8b")

    df.to_csv(plot_filename.split(".")[0] + ".csv")

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(x="Feature", y="Coefficient", data=df, palette=df["Color"])
    plt.ylim([-ymax, ymax])
    plt.title(plot_title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filename)
    print("Coefficients plot saved at {}".format(plot_filename))
    pass


def plot_confusion_matrix(filename=None, config_file="config.yml"):
    """Plots confusion matrix.

    Args:
        filename (str, optional): Desired filename for plot. Defaults to None.
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "Confusion Matrix | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_confusion_matrix.png".format(
        now, config["RUN_NAME"]
    )

    if filename == None:
        filename = config["REPORT"]

    df = pd.read_csv(filename)
    if config["TEST"] == True:
        df = df.loc[df.group == "TEST"].copy()
    y = df["target"]
    y_pred = df["prediction"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y, y_pred), cmap="Greens", annot=True, fmt="d")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_filename)
    print("Confusion matrix plot saved at {}".format(plot_filename))
    pass


def plot_pca(config_file="config.yml"):
    """Creates scatterplot of first two principal components.
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "PCA Scatterplot | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_PCA.png".format(now, config["RUN_NAME"])

    features = get_feature_names(k="auto", config_file=config_file)
    X, y = load_data(config["MAIN"], config_file=config_file)

    pca = PCA(n_components=2, random_state=88)
    components = pca.fit_transform(X[features])

    datamap = pd.concat([y, pd.DataFrame(components)], axis=1)
    datamap.columns = ["Group", "First Component", "Second Component"]
    datamap.to_csv(plot_filename[:-4] + ".csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    sns.scatterplot(
        x="First Component",
        y="Second Component",
        hue="Group",
        data=datamap,
        palette=["#7cfc00", "#551a8b"],
        ax=ax,
    )
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_filename)
    pass


def plot_box(filename=None, config_file="config.yml"):
    """Creates boxplots of predicted probabilites for each sample, grouped by true labels.

    Args:
        filename ([type], optional): Desired filename for plot. Defaults to None.
    """
    now = str(datetime.now().strftime("%Y%m%d"))
    config = load_config(config_file)

    plot_title = "Predicted Probability by Group | {} | {}".format(
        config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
    )
    plot_filename = "../output/{}_{}_boxplots.png".format(now, config["RUN_NAME"])

    if filename == None:
        filename = config["REPORT"]

    df = pd.read_csv(filename)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        x="group",
        y="probability",
        hue="target",
        data=df,
        palette=["plum", "palegreen"],
        hue_order=config["CLASSES"],
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        x="group",
        y="probability",
        hue="target",
        data=df,
        color="black",
        dodge=True,
        hue_order=config["CLASSES"],
        ax=ax,
    )
    plt.axhline(0.5, linestyle="--", color="black")
    plt.axvline(0.5, linestyle="--", color="black")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Predicted Probability - {}".format(config["CLASSES"][1]))
    plt.xlabel("Group")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(1.025, 0.5))
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_filename)
    pass


#####################
# UTILITY FUNCTIONS #
#####################


def get_feature_names(k="auto", selector_object=None, config_file="config.yml"):
    """Retrieves feature names from selection pipeline.

    Args:
        k (int, optional): Size of featureset to retrieve. Defaults to "auto".
        selector_object (sklearn pipeline, optional): Pipeline object. Defaults to None.

    Returns:
        [type]: [description]
    """
    config = load_config(config_file)
    k = self.best_k

    if selector_object == None:
        selector = joblib.load(config["SELECTOR"])
    else:
        selector = selector_object
    steps = {}
    for name, step in selector.steps:
        steps[name] = step

    X, _ = load_data(config["MAIN"], config_file=config_file)
    feature_names = pd.Series(X.columns)
    if "correlation_filter" in steps.keys():
        feature_names = pd.Series(steps["correlation_filter"].feature_names)
    if "mutual_info" in steps.keys():
        kbest_idx = steps["mutual_info"].get_support()
        feature_names = feature_names[kbest_idx].reset_index(drop=True)

    sffs_idx = list(steps["sffs"].get_metric_dict()[k]["feature_idx"])

    selected_features = feature_names[sffs_idx]
    return selected_features


def get_coefficients(feature_names, config_file="config.yml"):
    """Retrieve logistic regression coefficients for specified featureset.
    This function supports the coefficient bar graph.

    Args:
        feature_names (list): Names of selected features.

    Returns:
        coefficients (list): Coefficients.
    """
    config = load_config(config_file)
    X, y = load_data(config["MAIN"], config_file=config_file)
    classifier = build_classifier(config_file=config_file)
    print("Calculating coefficients...")
    classifier.fit(X[feature_names], y)
    coefficients = classifier.steps[-1][1].coef_[0]
    return coefficients


def process_pval(p):
    """Rounds p-value to three decimal places, or returns a string if it is below .001"""
    if p < 0.001:
        return "< 0.001"
    else:
        return str(round(p, 3))


###########################
# CLIFF'S DELTA FUNCTIONS #
###########################


def cliffsDelta(lst1, lst2, **dull):

    """Returns delta and pipeline if there are more than 'dull' differences"""
    if not dull:
        dull = {
            "small": 0.147,
            "medium": 0.33,
            "large": 0.474,
        }  # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j * repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j) * repeats
    d = (more - less) / (m * n)
    size = lookup_size(d, dull)
    return d, size


def lookup_size(delta: float, dull: dict) -> str:
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta < dull["small"]:
        return "negligible"
    if dull["small"] <= delta < dull["medium"]:
        return "small"
    if dull["medium"] <= delta < dull["large"]:
        return "medium"
    if delta >= dull["large"]:
        return "large"


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two
