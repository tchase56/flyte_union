from itertools import product
from typing import Any, Optional
from functools import partial
import operator

import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

from flytekit import task, workflow, map_task
from flytekit import ImageSpec, Deck


sklearn_image_spec = ImageSpec(
    name='wine',
    packages=["scikit-learn==1.5.1", "pandas==2.2.2", 'pyarrow', 'fastparquet', 'matplotlib', 'plotly'],
    registry='us-east4-docker.pkg.dev/union-ai-poc/fbin-union-ai-poc-docker',
    base_image="ghcr.io/flyteorg/flytekit:py3.11-latest"
)

cache_version = "cache-v1"

@task(container_image=sklearn_image_spec, cache=True, cache_version=cache_version)
def get_dataframe() -> pd.DataFrame:
    """
    Retrieves a pandas DataFrame containing wine data.
    Returns:
        pd.DataFrame: A DataFrame containing wine data.
    """
    
    return load_wine(as_frame=True).frame

@task(container_image=sklearn_image_spec, cache=True, cache_version=cache_version)
def create_search_grid(grid: dict[str, list[Optional[Any]]]) -> list[dict[str, Optional[Any]]]:
    """
    Generate a search grid based on the given dictionary of lists.
    Args:
        grid (dict[str, list[Optional[Any]]]): A dictionary where the keys represent the parameter names and the values are lists of possible values for each parameter.
    Returns:
        list[dict[str, Optional[Any]]]: Each possible permutation of the hyperparameter choices.
    """
    
    
    product_values = product(*[v if isinstance(v, (list, tuple)) else [v] for v in grid.values()])
    
    return [dict(zip(grid.keys(), values)) for values in product_values]

def split(dataframe: pd.DataFrame, test_size: float=0.25) -> tuple[pd.DataFrame, ...]:
    """
    Split the given dataframe into training and testing sets.
    Parameters:
        dataframe (pd.DataFrame): The input dataframe.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.25.
    Returns:
        tuple[pd.DataFrame, ...]: A tuple containing the training and testing sets.
    """
    
    
    targets = dataframe["target"]
    
    return train_test_split(
        dataframe.drop(columns = ["target"]),
        targets,
        test_size=test_size,
        stratify=targets,
        random_state=42
    )


@task(container_image=sklearn_image_spec, cache=True, cache_version=cache_version, enable_deck=True)
def train_model(
    dataframe: pd.DataFrame,
    hyperparameters: dict[str, Optional[Any]]
) -> RandomForestClassifier:
    """
    Trains a random forest classifier model using the given dataframe and hyperparameters.
    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing the training data.
        hyperparameters (dict[str, Optional[Any]]): The hyperparameters for the random forest classifier model.
    Returns:
        RandomForestClassifier: The trained random forest classifier model.
    """
    
    X_train, X_test, y_train, y_test = split(dataframe)

    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)

    return model

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots a confusion matrix based on the true labels and predicted labels.
    Parameters:
    - y_true (array-like): The true labels.
    - y_pred (array-like): The predicted labels.
    Returns:
    - fig (plotly.graph_objects.Figure): The confusion matrix plot.
    """
    
    
    array = confusion_matrix(y_true, y_pred)
    
    labels = y_true.unique().tolist()

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in array.tolist()]

    # set up figure 
    fig = ff.create_annotated_heatmap(array, x=labels, y=labels, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(
        title_text='Confusion Matrix',
        xaxis = dict(title='Predicted Label'),
        yaxis = dict(title='Actual Label')
    )

    # add custom xaxis title
    fig.add_annotation(dict(
        font=dict(color="black",size=14),
        x=0.5,
        y=-0.15,
        showarrow=False,
        text="Predicted value",
        xref="paper",
        yref="paper",
    ))

    # add custom yaxis title
    fig.add_annotation(dict(
        font=dict(color="black",size=14),
        x=-0.35,
        y=0.5,
        showarrow=False,
        text="Real value",
        textangle=-90,
        xref="paper",
        yref="paper"
    ))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    
    return fig

@task(container_image=sklearn_image_spec, cache=True, cache_version=cache_version, enable_deck=True)
def compare_model_results(
    dataframe: pd.DataFrame,
    models: list[RandomForestClassifier],
    hyperparameters: list[dict[str, Optional[Any]]],
) -> RandomForestClassifier:
    """
    Compares the results of different models on a given dataframe using specified hyperparameters.
    Args:
        dataframe (pd.DataFrame): The input dataframe.
        models (list[RandomForestClassifier]): List of models to compare.
        hyperparameters (list[dict[str, Optional[Any]]]): List of hyperparameters for each model.
    Returns:
        RandomForestClassifier: The best performing model based on F1 score.
    """
    
    X_train, X_test, y_train, y_test = split(dataframe)
    
    scores: list[float] = []
    
    for model in models:
        
        yhat = model.predict(X_test)
        
        score = f1_score(y_pred=yhat, y_true=y_test, average="macro")
        
        scores.append(score)
    
    which_best, _ = max(enumerate(scores), key=operator.itemgetter(1))
    
    df = pd.DataFrame.from_records(hyperparameters)
    df['scores'] = scores

    fig = px.parallel_coordinates(
        df, color="scores",
        dimensions=hyperparameters[0].keys(),
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2
    )

    fig.update_layout(
        title_text='Parellel Coordinates',
        xaxis = dict(title='Hyperparameter Names'),
        yaxis = dict(title='Hyperparameter Value')
    )

    Deck("Parallel Coordinates", plotly.io.to_html(fig))
    
    return models[which_best]


@task(container_image=sklearn_image_spec, cache=True, cache_version=cache_version, enable_deck=True)
def analyze_model(
    model: RandomForestClassifier,
    dataframe: pd.DataFrame,
) -> None:
    """
    Analyzes the performance of a RandomForestClassifier model on a given dataframe.
    Parameters:
        model (RandomForestClassifier): The trained RandomForestClassifier model.
        dataframe (pd.DataFrame): The dataframe containing the data for analysis.
    Returns:
        None
    """
    
    X_train, X_test, y_train, y_test = split(dataframe)
    
    yhat = model.predict(X_test)
    
    confusion_matrix_plot = plot_confusion_matrix(y_true=y_test, y_pred=yhat)

    Deck("Confusion Matrix", plotly.io.to_html(confusion_matrix_plot))
    

@workflow
def training_workflow(grid: dict[str, list[Optional[Any]]] = {
    'max_depth': [10, 50, 100],
    'max_features': [None, 'sqrt'],
    'n_estimators': [100, 2000]
}) -> RandomForestClassifier:
    """
    Executes the training workflow for a random forest classifier.
    Args:
        grid (dict[str, list[Optional[Any]]], optional): The grid of hyperparameters to search over. Defaults to a predefined grid.
    Returns:
        RandomForestClassifier: The best trained random forest classifier model.
    """

    dataframe = get_dataframe()
    
    hyperparameters = create_search_grid(grid=grid)

    models = map_task(partial(train_model, dataframe=dataframe))(hyperparameters=hyperparameters)
    
    best_model = compare_model_results(models=models, hyperparameters=hyperparameters, dataframe=dataframe)

    analyze_model(model=best_model, dataframe=dataframe)
    
    return best_model