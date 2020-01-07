import click
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

from model import robust_sffs

np.random.seed(88)

RF = RandomForestClassifier(
    n_jobs=-1, n_estimators=100, random_state=88
)
LR = LogisticRegression(solver="lbfgs", multi_class="auto")
BRF = BalancedRandomForestClassifier(n_jobs=-1, random_state=88)

@click.command()
@click.option("--command", help="The command to execute.")
@click.option("--config", help="The command to execute.")
@click.option("--clf", help="The command to execute.")
@click.option("--joblib_file", help="The command to execute.")
def interface(command, config, clf, joblib_file):
    if command == "select":
        if clf == "LR":
            pipe = robust_sffs(LR)
        if clf == "RF":
            pipe = robust_sffs(RF)
        pipe.load_config(config)
        pipe.load_data(pipe.config["MAIN"])
        pipe.select_features_cv()
        pipe.plot_sffs()
        joblib.dump(pipe, pipe.config["SELECTOR"])
    if command == "permtest":
        pipe = joblib.load(joblib_file)
        pipe.permutation_test_cv(k=3)
        pipe.plot_permtest()
        joblib.dump(pipe, pipe.config["PERMUTATION_TEST"])
    else:
        click.echo(
            'You must specify a command! Options: ["select", "permtest", "evaluate"]'
        )


if __name__ == "__main__":
    interface()
