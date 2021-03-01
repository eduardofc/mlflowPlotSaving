import mlflow
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


def read_data():
    data = load_breast_cancer(as_frame=True)
    features = data.data
    target = data.target
    return features, target


if __name__ == '__main__':
    features, target = read_data()

    sns.histplot(features['mean radius'])
    plt.savefig('sample.png')

    with mlflow.start_run():
        mlflow.log_artifact("./sample.png")

