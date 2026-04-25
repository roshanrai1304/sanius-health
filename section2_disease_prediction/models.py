from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, StackingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from . import config


def get_models():
    lr = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                         random_state=config.RANDOM_SEED, eval_metric="logloss")

    return [
        {"name": "Logistic Regression", "model": lr},
        {"name": "Random Forest", "model": rf},
        {"name": "XGBoost", "model": xgb},
        {"name": "SVM",
         "model": SVC(kernel="rbf", probability=True, random_state=config.RANDOM_SEED)},
        {"name": "KNN",
         "model": KNeighborsClassifier(n_neighbors=5)},
        {"name": "MLP",
         "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                random_state=config.RANDOM_SEED)},
        {"name": "Gradient Boosting",
         "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                             max_depth=5, random_state=config.RANDOM_SEED)},
        {"name": "AdaBoost",
         "model": AdaBoostClassifier(n_estimators=100, learning_rate=0.5,
                                     random_state=config.RANDOM_SEED)},
        {"name": "Bagging",
         "model": BaggingClassifier(n_estimators=50, random_state=config.RANDOM_SEED)},
        {"name": "Voting (Soft)",
         "model": VotingClassifier(
             estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
             voting="soft",
         )},
        {"name": "Stacking",
         "model": StackingClassifier(
             estimators=[("rf", rf), ("xgb", xgb)],
             final_estimator=LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
             cv=5,
         )},
    ]
