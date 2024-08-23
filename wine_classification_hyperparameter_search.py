import base64
import io
import html
from textwrap import dedent
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)
from flytekit import task, workflow
from typing import List, Tuple, Dict
from flytekit import ImageSpec, Deck, current_context
# import mlflow
# from flytekitplugins.mlflow import mlflow_autolog
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt


sklearn_image_spec = ImageSpec(
    name='wine',
    packages=["scikit-learn==1.5.1", "pandas==2.2.2", 'pyarrow', 'fastparquet', 'matplotlib'],
    registry='us-east4-docker.pkg.dev/union-ai-poc/fbin-union-ai-poc-docker',
    base_image="ghcr.io/flyteorg/flytekit:py3.11-latest"
)

@task(container_image=sklearn_image_spec)
def get_data() -> pd.DataFrame:
    """Get the wine dataset
    
    Parameters:
        None

    Returns:
        (pd.DataFrame): The wine dataset
    """
    # Load Data 
    data = load_wine(as_frame=True).frame

    return data

@task(container_image=sklearn_image_spec)
def process_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and testing sets using stratified sampling
    
    Parameters:
        data (pd.DataFrame): The wine dataset

    Returns:
        Tuple Containing:
            (pd.DataFrame): The training data
            (pd.DataFrame): The testing data
            (pd.DataFrame): The training target
            (pd.DataFrame): The testing target
    """

    # Split Data
    X_data = data.drop(columns = ["target"])
    y_data = data[["target"]]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, stratify=y_data, random_state=42)

    return X_train, X_test, y_train, y_test
    

@task(enable_deck=True, container_image=sklearn_image_spec)
# @mlflow_autolog(framework=mlflow.sklearn)
def hyperparameter_search(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> RandomForestClassifier:
    """
    Do hyperparameter tuning on a RandomForestClassifier and then display the results of the best model

    Parameters:
        X_train (pd.DataFrame): The training data
        X_test (pd.DataFrame): The testing data
        y_train (pd.DataFrame): The training target
        y_test (pd.DataFrame): The testing target

    Returns:
        (RandomForestClassifier): The best model
    """

    # Hyperparameter Search
    params = {
        'max_depth': [10, 50, 100],
        'max_features': [None, 'sqrt'],
        'n_estimators': [100, 1000, 2000]
    }
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid=params, scoring='f1_macro')  
    grid_search.fit(X_train, y_train['target'])

    _create_flytedeck(grid_search, X_train, X_test, y_train, y_test)

    return grid_search.best_estimator_

@workflow
def training_workflow() -> RandomForestClassifier:
    """
    Create workflow DAG (composed of tasks) to train a RandomForestClassifier on the wine dataset

    Parameters:
        None

    Returns:
        (RandomForestClassifier): The best model
    """
    # raise Exception("This is a test")
    data = get_data()
    X_train, X_test, y_train, y_test = process_data(data=data)

    best_model = hyperparameter_search(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    return best_model

def _convert_fig_into_html(fig: mpl.figure.Figure) -> str:
    """
    Convert a matplotlib figure into an HTML image string

    Parameters:
        fig (mpl.figure.Figure): The figure to convert

    Returns:
        (str): The HTML image string
    """
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_base64 = base64.b64encode(img_buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'

def _create_flytedeck(
        grid_search: GridSearchCV,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame
) -> None:
    """
    Create additional charts and plots in the flyte deck

    Parameters:
        grid_search (RandomForestClassifier): The best model
        X_train (pd.DataFrame): The training data
        X_test (pd.DataFrame): The testing data
        y_train (pd.DataFrame): The training target
        y_test (pd.DataFrame): The testing target

    Returns:
        None
    """

    # Evaluate the Best Model and add to the flyte deck
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    experiment_results = pd.DataFrame(grid_search.cv_results_)
    pd.DataFrame(grid_search.cv_results_)

    ############################
    # Metrics of Best Model 
    ############################
    metrics_deck = Deck("Metrics")

    # train
    fig, ax = plt.subplots()
    ax.set_title("Confusion Matrix: Train")
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=ax)
    metrics_deck.append(_convert_fig_into_html(fig))
    class_report = classification_report(y_train, y_train_pred)
    print("Classification report: Train")
    print(class_report)
    report = html.escape(class_report)
    html_report = dedent(
        f"""\
    <h2>Classification report: Train</h2>
    <pre>{report}</pre>"""
    )
    metrics_deck.append(html_report)

    # test
    fig2, ax2 = plt.subplots()
    ax2.set_title("Confusion Matrix: Test")
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax2)
    metrics_deck.append(_convert_fig_into_html(fig2))
    class_report2 = classification_report(y_test, y_test_pred)
    print("Classification report: Test")
    print(class_report2)
    report2 = html.escape(class_report2)
    html_report2 = dedent(
        f"""\
    <h2>Classification report: Test</h2>
    <pre>{report2}</pre>"""
    )
    metrics_deck.append(html_report2)
    ctx = current_context()
    ctx.decks.insert(0, metrics_deck)

    ############################
    # Experiment Results
    ############################
    experiment_deck = Deck("Experiment Results")
    print("Experiment Results")
    print(experiment_results)
    report = html.escape(experiment_results)
    html_report3 = dedent(
        f"""\
    <h2>Experiment Results</h2>
    <pre>{experiment_results}</pre>"""
    )
    experiment_deck.append(html_report3)
    ctx.decks.insert(0, experiment_deck)


if __name__ == "__main__":
    training_workflow()