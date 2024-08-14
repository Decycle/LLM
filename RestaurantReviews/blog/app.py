from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_latex as dl
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import time

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template("sketchy")
external_stylesheets = [
    dbc.themes.SKETCHY,
    dbc.icons.FONT_AWESOME,
    dbc_css,
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

df = pd.read_csv("../models/yelp_restaurant_review_labelled.csv")

categories = [
    "FOOD",
    "LOCATION",
    "ATOMSPHERE",
    "SERVICE",
    "PRICE",
    "MENU",
    "SPEED",
]


def draw_graph(threshold=0.1):
    random_entry = df.sample()
    for category in categories:
        usefulness = f"{category}_usefulness"
        score = f"{category}_score"
        random_entry.loc[random_entry[usefulness] < threshold, score] = 0
        random_entry.drop(columns=[usefulness], inplace=True)
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.H4("Review:", className="card-title"),
                    random_entry["text"].values[0],
                ],
            ),
            dbc.CardBody(
                [
                    dcc.Graph(
                        figure=px.bar(
                            x=categories,
                            y=[
                                random_entry[f"{category}_score"].values[0]
                                for category in categories
                            ],
                            title="Scores",
                            labels={"x": "Category", "y": "Score"},
                            range_y=[1, 5],
                        ),
                    )
                ]
            ),
        ]
    )


app.layout = dbc.Container(
    [
        html.Title("Restaurant Reviews"),
        html.H1(
            [
                "Decoders",
                html.I(className="fa-solid fa-screwdriver-wrench fs-2"),
            ],
            className="fw-bold",
        ),
        html.P(
            """
        With the rise of machine learning transformers such as large language models, we can utilize them to help us understand various aspects of the reviews.
        """
        ),
        dl.DashLatex(
            r"""
            We will use the probability output of the model for the last token.
            We will record the logits for each of the rating number, as well as the probability of the model saying "NOT".
            Let the logits for rating number $i$ be $l_i$, the score can be calculated as
            """,
        ),
        dl.DashLatex(
            r"$$s = \frac{\sum_{i=1}^5 i \exp(l_i)}{\sum_{i=1}^5 \exp(l_i)}$$",
            displayMode=True,
        ),
        dl.DashLatex(
            r"""
            for that category.

            If the probablity of the model saying "NOT" is very high, that means the model thinks the category is not mentioned in the review. In that case we should discard the rating.

            In particular, denote the logits for predicting NOT as $l_n$, we can extract a "usefulness" parameter $u$
            """
        ),
        dl.DashLatex(
            r"""
            $$u = \frac{\sum_{i=1}^5 \exp(l_i)}{exp(l_n) + \sum_{i=1}^5 \exp(l_i)}$$
            """,
            displayMode=True,
        ),
        html.P("Click the button below to see a new review from the training data!"),
        dbc.Button(
            [
                html.I(className="fas fa-sync-alt"),
                " New Review",
            ],
            id="refresh",
            n_clicks=0,
            className="mb-4",
        ),
        dbc.Col(
            children=[
                draw_graph(),
            ],
            id="bar-graphs",
        ),
    ],
    className="w-50 dbc",
    fluid=True,
)


@callback(
    Output("bar-graphs", "children"),
    Input("refresh", "n_clicks"),
)
def update_graphs(n_clicks):
    return draw_graph()


if __name__ == "__main__":
    app.run(debug=True)
