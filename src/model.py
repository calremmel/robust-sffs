from __future__ import division

import json
from datetime import datetime

import attr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from functions import CorrelationThreshhold, split_dataset

np.random.seed(88)


@attr.s
class robust_sffs(object):
    clf = attr.ib()

    def load_config(self, filename):
        """Loads a configuration file from a  specified location.
        
        Args:
            filename (str, optional): Config file path. Defaults to "config.yml".
        
        Returns:
            config (dict): Dictionary of config settings.
        """
        with open(filename, "r") as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)

        if config["RUN_DATE"] == "auto":
            now = str(datetime.now().strftime("%Y%m%d"))
        else:
            now = config["RUN_DATE"]

        config["SELECTOR"] = "../output/{}_{}_selector.joblib".format(
            now, config["RUN_NAME"]
        )
        config["PERMUTATION_TEST"] = "../output/{}_{}_permtest.joblib".format(
            now, config["RUN_NAME"]
        )
        config["REPORT"] = "../output/{}_{}_classifier_report.csv".format(
            now, config["RUN_NAME"]
        )
        self.config = config

    def load_data(self, filename, meta_cols=False, meta_return=False):
        """Loads data from .csv file, optionally removing columns 
        that contain metadata better used for grouping than for modeling.
        
        Args:
            filename (str): Path to .csv file containing data.
            meta_cols (bool, optional): True if there are metadata columns to remove. Defaults to False.
            meta_return (bool, optional): True to return the metadata columns as a separate DataFrame. Defaults to False.
        """
        X = pd.read_csv(filename)
        X = X.dropna(axis=1)
        y = X.pop(self.config["TARGET"])

        # Drop columns containing metadata
        if meta_cols == False:
            if self.config["META_COLS"] != "N":
                if len(self.config["META_COLS"]) == 1:
                    meta = X.pop(self.config["META_COLS"][0])
                else:
                    meta = X[self.config["META_COLS"]].copy()
                    X = X.drop(self.config["META_COLS"], axis=1).copy()
        self.X = X
        self.y = y
        if meta_return == True and self.config["META_COLS"] != "N":
            self.meta = meta

    def get_steps(self, type):
        """Loads pipeline steps from configuration settings.
        
        Args:
            type (str): Type of pipeline to construct ("selector" or "classifier").
        
        Returns:
            processed_steps (list): List of pipeline steps.
        """        
        if type == "selector":
            steps = [
                ("correlation_filter", CorrelationThreshhold()),
                ("logscale", FunctionTransformer(np.log1p, validate=True)),
                ("normalize", StandardScaler()),
                ("mutual_info", SelectKBest(score_func=mutual_info_classif, k=10)),
            ]
        elif type == "classifier":
            steps = [
                ("logscale", FunctionTransformer(np.log1p, validate=True)),
                ("normalize", StandardScaler()),
            ]

        processed_steps = []

        for step in steps:
            key = step[0].upper()
            if self.config[key]["USE"] == "Y":
                processed_steps.append(step)

        return processed_steps

    def build_selector(self):
        "Creates a feature selection pipeline."
        # Instantiate sequential forward floating selection object
        self.max_features = self.config["MAX_K"]
        sffs = SFS(
            self.clf,
            k_features=self.max_features,
            forward=True,
            floating=True,
            verbose=0,
            scoring="accuracy",
            cv=0,
            n_jobs=-1,
        )

        # Instantiate pipeline steps

        steps = self.get_steps("selector")
        steps += [("sffs", sffs)]
        # Instantiate feature selection pipeline
        selector = Pipeline(steps=steps)

        self.selector = selector

    def build_classifier(self):
        "Creates a classification pipeline, using settings from the feature selection pipeline."
        # Instantiate pipeline steps
        steps = self.get_steps("classifier")
        steps += [("clf", self.clf)]

        # Instantiate classification pipeline
        classifier = Pipeline(steps=steps)

        self.classifier = classifier

    def build_transformer(self):
        "Creates a data preprocessing pipeline."
        # Instantiate pipeline steps
        steps = [
            ("logscale", FunctionTransformer(np.log1p, validate=True)),
            ("normalize", StandardScaler()),
        ]

        # Instantiate classification pipeline
        transformer = Pipeline(steps=steps)

        self.transformer = transformer

    def get_parsimonious_k(self):
        "Finds the smallest number of features such that the accuracy is no lower than 1 sdev lower than the maximum accuracy."
        metric_data = self.eval_byfold(self.selector_dict)
        is_true_and_test = (metric_data["run"] == "True") & (
            metric_data["fold"] == "Test"
        )
        mean_accs_by_fsize = (
            metric_data.loc[is_true_and_test].groupby("num_feats").mean().reset_index()
        )
        max_acc = np.max(mean_accs_by_fsize["accuracy"])
        stdev = np.std(mean_accs_by_fsize["accuracy"])
        acc_parsimonious = max_acc - stdev
        fsize_parsimonious = mean_accs_by_fsize.loc[
            mean_accs_by_fsize["accuracy"] >= acc_parsimonious, "num_feats"
        ].min()
        self.best_k = fsize_parsimonious

    def select_features_cv(self):
        "Runs a feature selection pipeline."
        self.cv = list(
            RepeatedStratifiedKFold(
                n_splits=self.config["CV"]["FOLDS"],
                n_repeats=self.config["CV"]["REPEATS"],
                random_state=88,
            ).split(self.X, self.y)
        )
        self.build_selector()
        self.build_classifier()

        selector_dict = {x: {} for x in range(1, len(self.cv) + 1)}
        print("Searching for features...")
        for j, split in enumerate(self.cv):
            print("Fold #{}".format(j + 1))
            X_temp = self.X.iloc[split[0], :]
            y_temp = self.y.iloc[split[0]]
            selector_temp = clone(self.selector)
            selector_temp.fit(X_temp, y_temp)
            selector_dict[j + 1]["data"] = {
                "X_train": X_temp,
                "y_train": y_temp,
                "X_test": self.X.iloc[split[1], :],
                "y_test": self.y.iloc[split[1]],
            }
            selector_dict[j + 1]["metric_dict"] = selector_temp[-1].get_metric_dict()
            for k in selector_dict[j + 1]["metric_dict"].values():
                feature_idx = list(k["feature_idx"])
                k["feature_names"] = list(
                    selector_dict[j + 1]["data"]["X_train"].iloc[:, feature_idx].columns
                )
                classifier_temp = clone(self.classifier)
                classifier_temp.fit(
                    selector_dict[j + 1]["data"]["X_train"].iloc[:, feature_idx],
                    selector_dict[j + 1]["data"]["y_train"],
                )
                y_pred = classifier_temp.predict(
                    selector_dict[j + 1]["data"]["X_test"].iloc[:, feature_idx]
                )
                y_prob = classifier_temp.predict_proba(
                    selector_dict[j + 1]["data"]["X_test"].iloc[:, feature_idx]
                )
                test_score = accuracy_score(
                    selector_dict[j + 1]["data"]["y_test"], y_pred
                )
                k["y_pred"] = y_pred
                k["y_prob"] = y_prob
                k["test_score"] = test_score
        self.selector_dict = selector_dict
        self.get_parsimonious_k()

    def permutation_test_cv(self, n_shuffles=10, k=2):
        """Runs feature selection pipeline with permuted labels multiple times.
        
        Args:
            n_shuffles (int, optional): Number of times to shuffle labels. Defaults to 10.
            k (int, optional): Maximum featureset size for permutation runs. Defaults to 2.
        """
        permutation_dict = {x: {} for x in range(1, n_shuffles + 1)}
        print("Running permutation test...")
        for i in range(n_shuffles):
            for j, split in enumerate(self.cv):
                print("Permutation #{}, Fold #{}".format(i + 1, j + 1))
                X_temp = self.X.iloc[split[0], :]
                y_perm = pd.Series(np.random.RandomState(seed=j).permutation(self.y))
                y_temp = y_perm.iloc[split[0]]
                selector_temp = clone(self.selector)
                permutation_dict[i + 1][j + 1] = {}
                permutation_dict[i + 1][j + 1]["data"] = {
                    "X_train": X_temp,
                    "y_train": y_temp,
                    "X_test": self.X.iloc[split[1], :],
                    "y_test": y_perm.iloc[split[1]],
                    "seed": j,
                }
                selector_temp.fit(X_temp, y_temp)
                permutation_dict[i + 1][j + 1]["metric_dict"] = selector_temp[
                    -1
                ].get_metric_dict()
                for k in permutation_dict[i + 1][j + 1]["metric_dict"].values():
                    feature_idx = list(k["feature_idx"])
                    k["feature_names"] = list(
                        permutation_dict[i + 1][j + 1]["data"]["X_train"]
                        .iloc[:, feature_idx]
                        .columns
                    )
                    classifier_temp = clone(self.classifier)
                    classifier_temp.fit(
                        permutation_dict[i + 1][j + 1]["data"]["X_train"].iloc[
                            :, feature_idx
                        ],
                        permutation_dict[i + 1][j + 1]["data"]["y_train"],
                    )
                    y_pred = classifier_temp.predict(
                        permutation_dict[i + 1][j + 1]["data"]["X_test"].iloc[
                            :, feature_idx
                        ]
                    )
                    y_prob = classifier_temp.predict_proba(
                        permutation_dict[i + 1][j + 1]["data"]["X_test"].iloc[
                            :, feature_idx
                        ]
                    )
                    test_score = accuracy_score(
                        permutation_dict[i + 1][j + 1]["data"]["y_test"], y_pred
                    )
                    k["y_pred"] = y_pred
                    k["y_prob"] = y_prob
                    k["test_score"] = test_score
        self.permutation_dict = permutation_dict

    def eval_byfold(self, metric_dict):
        """[summary]
        
        Args:
            metric_dict ([type]): [description]
        
        Returns:
            [type]: [description]
        
        Yields:
            [type]: [description]
        """
        folds, repeats = self.config["CV"]["FOLDS"], self.config["CV"]["REPEATS"]

        repeats_list = [
            y
            for x in range(1, repeats + 1)
            for y in [x] * (2 * self.max_features * folds)
        ]
        split_list = []
        fsize_list = []
        fold_list = []
        acc_list = []

        for split in metric_dict.keys():
            for num_feats in metric_dict[split]["metric_dict"].keys():
                train_acc = metric_dict[split]["metric_dict"][num_feats]["avg_score"]
                test_acc = metric_dict[split]["metric_dict"][num_feats]["test_score"]
                acc_list += [train_acc, test_acc]
                fold_list += ["Train", "Test"]
                fsize_list += [num_feats] * 2
                split_list += [split] * 2

        plot_data = pd.DataFrame(
            {
                "repeat": repeats_list,
                "split": split_list,
                "num_feats": fsize_list,
                "fold": fold_list,
                "accuracy": acc_list,
                "permutation": [0 for x in repeats_list],
                "run": ["True" for x in repeats_list],
            }
        )

        return plot_data

    def plot_sffs(self):
        """[summary]
        
        Returns:
            [type]: [description]
        
        Yields:
            [type]: [description]
        """
        now = str(datetime.now().strftime("%Y%m%d"))
        plot_title = "Accuracy vs Number of Features | {} | {}".format(
            self.config["ESTIMATOR"]["CODE"], self.config["RUN_NAME"]
        )
        plot_filename = "../output/{}_{}_sffs_plot.png".format(
            now, self.config["RUN_NAME"]
        )
        output_filename = "../output/{}_{}_sffs_plot.csv".format(
            now, self.config["RUN_NAME"]
        )
        plot_data = self.eval_byfold(self.selector_dict)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.lineplot(
            x="num_feats", y="accuracy", hue="fold", data=plot_data, ci="sd", alpha=0.5
        )
        plt.ylim([-0.1, 1.1])
        plt.ylabel("Accuracy")
        plt.xlabel("Number of Features")
        plt.title(plot_title)
        plt.xticks(list(range(1, 11)))
        plt.grid(axis="x")
        plt.axhline(0.5, color="r", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_filename)
        plot_data.to_csv(output_filename, index=False)

    def plot_permtest(self, k=None):
        """[summary]
        
        Args:
            k ([type], optional): [description]. Defaults to None.
        """
        now = str(datetime.now().strftime("%Y%m%d"))
        plot_title = "Accuracy vs Run | {} | {}".format(
            self.config["ESTIMATOR"]["CODE"], self.config["RUN_NAME"]
        )
        plot_filename = "../output/{}_{}_permutation_violins.png".format(
            now, self.config["RUN_NAME"]
        )
        if k is None:
            k = self.best_k
        perm_eval_dicts = []
        for x in self.permutation_dict.values():
            perm_eval_dicts.append(self.eval_byfold(x))

        for x in range(len(perm_eval_dicts)):
            perm_eval_dicts[x]["permutation"] = x + 1
            perm_eval_dicts[x]["run"] = "Permuted"

        all_eval_dicts = pd.concat(
            perm_eval_dicts + [self.eval_byfold(self.selector_dict)]
        )

        mean_by_repeat = (
            all_eval_dicts.groupby(["run", "repeat", "num_feats", "fold"])
            .mean()
            .reset_index()
            .drop(["split", "permutation"], axis=1)
        )

        mean_by_repeat

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.violinplot(
            x="run",
            y="accuracy",
            hue="fold",
            data=mean_by_repeat.loc[mean_by_repeat.num_feats == k],
            order=["True", "Permuted"],
            hue_order=["Train", "Test"],
        )
        plt.title(plot_title)
        plt.ylim([-0.1, 1.1])
        plt.axhline(0.5, color="r", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_filename)


# def permtest_evaluate(config_file="config.yml", features=None):
#     """[summary]

#     Args:
#         config_file (str, optional): [description]. Defaults to "config.yml".
#         k (str, optional): [description]. Defaults to "auto".
#         features ([type], optional): [description]. Defaults to None.

#     Returns:
#         [type]: [description]

#     Yields:
#         [type]: [description]
#     """
#     config = load_config(config_file)
#     k = self.best_k

#     true = joblib.load(config["SELECTOR"])
#     permuted = joblib.load(config["PERMUTATION_TEST"])
#     # get true cv scores
#     if type(features) == list:
#         X, y = load_data(config["MAIN"], config_file=config_file)
#         classifier = build_classifier(config_file=config_file)
#         stratkfold = RepeatedStratifiedKFold(
#             n_splits=config["CV"]["FOLDS"],
#             n_repeats=config["CV"]["REPEATS"],
#             random_state=88,
#         ).split(X, y)
#         cv = list(stratkfold)
#         true_scores = cross_val_score(classifier, X[features], y, cv=cv)
#     else:
#         true_scores = list(true.steps[-1][1].get_metric_dict()[k]["cv_scores"])
#         true_scores = [
#             true_scores[i : i + config["CV"]["FOLDS"]]
#             for i in range(0, len(true_scores), config["CV"]["FOLDS"])
#         ]
#         true_scores = [np.mean(x) for x in true_scores]
#     # get cv scores from permutation test
#     permuted_scores = []
#     for i in permuted:
#         scores = list(permuted[i].steps[-1][1].get_metric_dict()[k]["cv_scores"])
#         permuted_scores.append(np.mean(scores))
#     plot_filename = plot_permutation_violins(
#         true_scores, permuted_scores, config_file=config_file
#     )
#     print("Permutation test plot saved at {}".format(plot_filename))
#     pass


# def output_predictions(config_file="config.yml", k="auto", features=None):
#     """Outputs a .csv file of predicted probabilities and labels, annotated with metadata if available.

#     Args:
#         k ([type], optional): [description]. Defaults to config['K_FEATURES'].
#         features ([type], optional): [description]. Defaults to None.

#     Returns:
#         report (DataFrame): DataFrame of predictions and probabilities.
#     """
#     config = load_config(config_file)
#     k = self.best_k
#     if type(features) != list:
#         features = get_feature_names(k, config_file=config_file)
#     if config["META_COLS"] != "N":
#         X, y, meta = load_data(
#             config["MAIN"], config_file=config_file, meta_return=True
#         )
#     else:
#         X, y = load_data(config["MAIN"], config_file=config_file)

#     classifier = build_classifier(config_file=config_file)
#     classifier.fit(X[features], y)
#     y_pred = classifier.predict(X[features])
#     y_prob = classifier.predict_proba(X[features])[:, 1]
#     group = ["MAIN" for x in y_pred]
#     report = pd.DataFrame(
#         {"group": group, "target": y, "prediction": y_pred, "probability": y_prob}
#     )
#     if config["META_COLS"] != "N":
#         report = pd.concat([meta, report], axis=1)

#     if config["TEST"]:
#         if config["META_COLS"] != "N":
#             Xt, yt, metat = load_data(
#                 config["TEST"], meta_return=True, config_file=config_file
#             )
#         else:
#             Xt, yt = load_data(config["TEST"], config_file=config_file)
#         yt_pred = classifier.predict(Xt[features])
#         yt_prob = classifier.predict_proba(Xt[features])[:, 1]
#         groupt = ["TEST" for x in yt_pred]
#         report_test = pd.DataFrame(
#             {
#                 "group": groupt,
#                 "target": yt,
#                 "prediction": yt_pred,
#                 "probability": yt_prob,
#             }
#         )
#         if config["META_COLS"] != "N":
#             report_test = pd.concat([metat, report_test], axis=1)
#         report = pd.concat([report, report_test], axis=0)
#     report.to_csv(config["REPORT"], index=False)
#     print("Predictions saved at {}".format(config["REPORT"]))
#     return report


# ######################
# # PLOTTING FUNCTIONS #
# ######################


# def plot_permutation_violins(
#     pipeline_scores, permuted_scores, config_file="config.yml"
# ):
#     """Create violin plots of true vs permuted pipeline runs.

#     Args:
#         pipeline_scores (list): List of accuracy scores, true labels.
#         permuted_scores (list): List of accuracy scores, permuted labels.

#     Returns:
#         plot_filename (str): Output file location.
#     """
#     now = str(datetime.now().strftime("%Y%m%d"))
#     config = load_config(config_file)

#     plot_title = "Accuracy vs Run | {} | {}".format(
#         config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
#     )
#     plot_filename = "../output/{}_{}_permutation_violins.png".format(
#         now, config["RUN_NAME"]
#     )

#     pipeline = pd.DataFrame({"Run": "True", "Accuracy": pipeline_scores})
#     permuted = pd.DataFrame({"Run": "Permuted", "Accuracy": permuted_scores})
#     for_plotting = pd.concat([pipeline, permuted], axis=0)

#     _, p = ttest_ind(pipeline_scores, permuted_scores)
#     d, _ = cliffsDelta(pipeline_scores, permuted_scores)
#     p = process_pval(p)
#     d = str(round(d, 3))

#     for_plotting["pval"] = p
#     for_plotting["cliffsdelta"] = d
#     for_plotting.to_csv(plot_filename.split(".")[0] + ".csv")
#     current_palette = sns.color_palette()

#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.violinplot(
#         x="Run",
#         y="Accuracy",
#         data=for_plotting,
#         bw="scott",
#         cut=0,
#         palette=[current_palette[1], current_palette[0]],
#         ax=ax,
#     )
#     plt.axhline(0.5, color="r", linestyle="--")
#     plt.axhline(np.mean(pipeline_scores), color="darkorange", linestyle="--")
#     plt.axhline(np.mean(permuted_scores), color="darkblue", linestyle="--")
#     plt.ylim([-0.1, 1.1])
#     plt.title(plot_title)
#     plt.text(
#         0.5,
#         0.1,
#         "P Value: " + p + "\nCliff's Delta: " + d,
#         horizontalalignment="center",
#     )
#     plt.tight_layout()
#     plt.savefig(plot_filename)
#     return plot_filename


# def plot_feature_search(
#     pipeline, X=None, y=None, X_test=None, y_test=None, config_file="config.yml"
# ):
#     """Plots CV accuracy vs number of features. Optionally includes test set for comparison.

#     Args:
#         pipeline (sklearn pipeline): Feature selection pipeline.
#         X (array, optional): Matrix of features. Defaults to None.
#         y (array, optional): Array of labels. Defaults to None.
#         X_test (array, optional): Matrix of features, test set. Defaults to None.
#         y_test (array, optional): Array of labels, test set. Defaults to None.

#     Returns:
#         [type]: [description]
#     """
#     now = str(datetime.now().strftime("%Y%m%d"))
#     config = load_config(config_file)
#     plot_title = "SFFS Accuracy vs Number of Features | {} | {}".format(
#         config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
#     )
#     search_object = pipeline.steps[-1][1]
#     clf = search_object.estimator

#     crossval_scores = pd.DataFrame(
#         {
#             key: val
#             for key, val in zip(
#                 search_object.get_metric_dict().keys(),
#                 [
#                     search_object.get_metric_dict()[x]["cv_scores"]
#                     for x in search_object.get_metric_dict().keys()
#                 ],
#             )
#         }
#     )
#     crossval_scores = crossval_scores.melt()

#     if type(X_test) == pd.core.frame.DataFrame:
#         test_index = []
#         test_accuracy = []
#         evaluator = build_classifier()
#         for x in search_object.get_metric_dict().keys():
#             test_features = get_feature_names(x, config_file=config_file)
#             print("Features for set {}:\n{}".format(x, test_features))
#             evaluator.fit(X[test_features], y)
#             predictions = evaluator.predict(X_test[test_features])
#             accuracy = accuracy_score(y_test, predictions)
#             test_index.append(x)
#             test_accuracy.append(accuracy)

#     fig, ax = plt.subplots(figsize=(18, 8))
#     sns.lineplot(
#         x="variable",
#         y="value",
#         data=crossval_scores,
#         ci="sd",
#         ax=ax,
#         label="Cross-Validation",
#     )
#     if type(X_test) == pd.core.frame.DataFrame:
#         plt.plot(test_index, test_accuracy, label="Test Accuracy")
#     plt.xlabel("Number of Features")
#     plt.ylabel("Accuracy")
#     plt.title(plot_title)

#     num_features = len(crossval_scores.variable.unique())
#     plt.xticks(list(range(1, num_features + 1)), rotation=90)
#     plt.ylim([-0.05, 1.15])
#     plt.legend(loc="upper left")
#     plt.grid()
#     plt.tight_layout()
#     plot_filename = "../output/{}_{}_sffs_plot.png".format(now, config["RUN_NAME"])
#     plt.savefig(plot_filename)
#     return plot_filename


# def plot_coefficients(k="auto", config_file="config.yml"):
#     """Plots coefficient bars. Logistic Regression only.

#     Args:
#         k (int, optional): Feature set size to plot. Defaults to config file setting.
#     """
#     now = str(datetime.now().strftime("%Y%m%d"))
#     config = load_config(config_file)

#     plot_title = "Coefficients | {} | {}".format(
#         config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
#     )
#     plot_filename = "../output/{}_{}_coefficients.png".format(now, config["RUN_NAME"])

#     features = get_feature_names(k, config_file=config_file)
#     coefficients = get_coefficients(features, config_file=config_file)

#     df = pd.DataFrame({"Feature": features, "Coefficient": coefficients})

#     largest_coefficient = max(np.abs(coefficients))
#     ymax = largest_coefficient + 0.10 * largest_coefficient

#     df = df.sort_values("Coefficient", ascending=False)
#     df["Color"] = df.Coefficient.apply(lambda x: "#7cfc00" if x >= 0 else "#551a8b")

#     df.to_csv(plot_filename.split(".")[0] + ".csv")

#     fig, ax = plt.subplots(figsize=(6, 8))
#     sns.barplot(x="Feature", y="Coefficient", data=df, palette=df["Color"])
#     plt.ylim([-ymax, ymax])
#     plt.title(plot_title)
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.savefig(plot_filename)
#     print("Coefficients plot saved at {}".format(plot_filename))
#     pass


# def plot_confusion_matrix(filename=None, config_file="config.yml"):
#     """Plots confusion matrix.

#     Args:
#         filename (str, optional): Desired filename for plot. Defaults to None.
#     """
#     now = str(datetime.now().strftime("%Y%m%d"))
#     config = load_config(config_file)

#     plot_title = "Confusion Matrix | {} | {}".format(
#         config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
#     )
#     plot_filename = "../output/{}_{}_confusion_matrix.png".format(
#         now, config["RUN_NAME"]
#     )

#     if filename == None:
#         filename = config["REPORT"]

#     df = pd.read_csv(filename)
#     if config["TEST"] == True:
#         df = df.loc[df.group == "TEST"].copy()
#     y = df["target"]
#     y_pred = df["prediction"]

#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.heatmap(confusion_matrix(y, y_pred), cmap="Greens", annot=True, fmt="d")
#     plt.ylabel("True")
#     plt.xlabel("Predicted")
#     plt.title(plot_title)
#     plt.tight_layout()
#     plt.savefig(plot_filename)
#     print("Confusion matrix plot saved at {}".format(plot_filename))
#     pass


# def plot_pca(config_file="config.yml"):
#     """Creates scatterplot of first two principal components.
#     """
#     now = str(datetime.now().strftime("%Y%m%d"))
#     config = load_config(config_file)

#     plot_title = "PCA Scatterplot | {} | {}".format(
#         config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
#     )
#     plot_filename = "../output/{}_{}_PCA.png".format(now, config["RUN_NAME"])

#     features = get_feature_names(k="auto", config_file=config_file)
#     X, y = load_data(config["MAIN"], config_file=config_file)

#     pca = PCA(n_components=2, random_state=88)
#     components = pca.fit_transform(X[features])

#     datamap = pd.concat([y, pd.DataFrame(components)], axis=1)
#     datamap.columns = ["Group", "First Component", "Second Component"]
#     datamap.to_csv(plot_filename[:-4] + ".csv", index=False)
#     fig, ax = plt.subplots(figsize=(7, 6.5))
#     sns.scatterplot(
#         x="First Component",
#         y="Second Component",
#         hue="Group",
#         data=datamap,
#         palette=["#7cfc00", "#551a8b"],
#         ax=ax,
#     )
#     plt.title(plot_title)
#     plt.tight_layout()
#     plt.savefig(plot_filename)
#     pass


# def plot_box(filename=None, config_file="config.yml"):
#     """Creates boxplots of predicted probabilites for each sample, grouped by true labels.

#     Args:
#         filename ([type], optional): Desired filename for plot. Defaults to None.
#     """
#     now = str(datetime.now().strftime("%Y%m%d"))
#     config = load_config(config_file)

#     plot_title = "Predicted Probability by Group | {} | {}".format(
#         config["ESTIMATOR"]["CODE"], config["RUN_NAME"]
#     )
#     plot_filename = "../output/{}_{}_boxplots.png".format(now, config["RUN_NAME"])

#     if filename == None:
#         filename = config["REPORT"]

#     df = pd.read_csv(filename)

#     fig, ax = plt.subplots(figsize=(9, 6))
#     sns.boxplot(
#         x="group",
#         y="probability",
#         hue="target",
#         data=df,
#         palette=["plum", "palegreen"],
#         hue_order=config["CLASSES"],
#         fliersize=0,
#         ax=ax,
#     )
#     sns.stripplot(
#         x="group",
#         y="probability",
#         hue="target",
#         data=df,
#         color="black",
#         dodge=True,
#         hue_order=config["CLASSES"],
#         ax=ax,
#     )
#     plt.axhline(0.5, linestyle="--", color="black")
#     plt.axvline(0.5, linestyle="--", color="black")
#     plt.ylim(-0.05, 1.05)
#     plt.ylabel("Predicted Probability - {}".format(config["CLASSES"][1]))
#     plt.xlabel("Group")
#     handles, labels = ax.get_legend_handles_labels()
#     plt.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(1.025, 0.5))
#     plt.title(plot_title)
#     plt.tight_layout()
#     plt.savefig(plot_filename)
#     pass


# #####################
# # UTILITY FUNCTIONS #
# #####################


# def get_feature_names(k="auto", selector_object=None, config_file="config.yml"):
#     """Retrieves feature names from selection pipeline.

#     Args:
#         k (int, optional): Size of featureset to retrieve. Defaults to "auto".
#         selector_object (sklearn pipeline, optional): Pipeline object. Defaults to None.

#     Returns:
#         [type]: [description]
#     """
#     config = load_config(config_file)
#     k = self.best_k

#     if selector_object == None:
#         selector = joblib.load(config["SELECTOR"])
#     else:
#         selector = selector_object
#     steps = {}
#     for name, step in selector.steps:
#         steps[name] = step

#     X, _ = load_data(config["MAIN"], config_file=config_file)
#     feature_names = pd.Series(X.columns)
#     if "correlation_filter" in steps.keys():
#         feature_names = pd.Series(steps["correlation_filter"].feature_names)
#     if "mutual_info" in steps.keys():
#         kbest_idx = steps["mutual_info"].get_support()
#         feature_names = feature_names[kbest_idx].reset_index(drop=True)

#     sffs_idx = list(steps["sffs"].get_metric_dict()[k]["feature_idx"])

#     selected_features = feature_names[sffs_idx]
#     return selected_features


# def get_coefficients(feature_names, config_file="config.yml"):
#     """Retrieve logistic regression coefficients for specified featureset.
#     This function supports the coefficient bar graph.

#     Args:
#         feature_names (list): Names of selected features.

#     Returns:
#         coefficients (list): Coefficients.
#     """
#     config = load_config(config_file)
#     X, y = load_data(config["MAIN"], config_file=config_file)
#     classifier = build_classifier(config_file=config_file)
#     print("Calculating coefficients...")
#     classifier.fit(X[feature_names], y)
#     coefficients = classifier.steps[-1][1].coef_[0]
#     return coefficients


# def process_pval(p):
#     """Rounds p-value to three decimal places, or returns a string if it is below .001"""
#     if p < 0.001:
#         return "< 0.001"
#     else:
#         return str(round(p, 3))


# ###########################
# # CLIFF'S DELTA FUNCTIONS #
# ###########################


# def cliffsDelta(lst1, lst2, **dull):

#     """Returns delta and pipeline if there are more than 'dull' differences"""
#     if not dull:
#         dull = {
#             "small": 0.147,
#             "medium": 0.33,
#             "large": 0.474,
#         }  # effect sizes from (Hess and Kromrey, 2004)
#     m, n = len(lst1), len(lst2)
#     lst2 = sorted(lst2)
#     j = more = less = 0
#     for repeats, x in runs(sorted(lst1)):
#         while j <= (n - 1) and lst2[j] < x:
#             j += 1
#         more += j * repeats
#         while j <= (n - 1) and lst2[j] == x:
#             j += 1
#         less += (n - j) * repeats
#     d = (more - less) / (m * n)
#     size = lookup_size(d, dull)
#     return d, size


# def lookup_size(delta: float, dull: dict) -> str:
#     """
#     :type delta: float
#     :type dull: dict, a dictionary of small, medium, large thresholds.
#     """
#     delta = abs(delta)
#     if delta < dull["small"]:
#         return "negligible"
#     if dull["small"] <= delta < dull["medium"]:
#         return "small"
#     if dull["medium"] <= delta < dull["large"]:
#         return "medium"
#     if delta >= dull["large"]:
#         return "large"


# def runs(lst):
#     """Iterator, chunks repeated values"""
#     for j, two in enumerate(lst):
#         if j == 0:
#             one, i = two, 0
#         if one != two:
#             yield j - i, one
#             i = j
#         one = two
#     yield j - i + 1, two
