
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import statsmodels.formula.api as smf
from graphs import *
import pandas as pd
from Plots.graphs_full import *
from Plots.graphs_slr import *


data = pd.read_csv("Data/masters_salary.csv")


me_fig = build_mixed_effects_figure()
me_pred_fig = build_predicted_vs_actual_figure(data)

slr_fig = graph_slr("Data/masters_salary.csv")
mlr_fig = graphs_full("Data/masters_salary.csv")


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def serve_layout():
    return html.Div(
        id="page-container",
        children=[
            # ---------- Header ----------
            dbc.Navbar(
                [
                    html.Div("Mixed Effects Models", className="navbar-brand title"),
                ],
                className="header",
                expand=True,
            ),

            # ---------- Body container ----------
            html.Div(
                className="layout-container",
                children=[
                    # Sidebar
                    html.Div(
                        [
                            html.H2("Contents", className="sidebar-title"),
                            html.Hr(),
                            dbc.Nav(
                                [
                                    dbc.NavLink("Introduction", href="#introduction"),
                                    dbc.NavLink("Main Content", href="#main"),
                                    dbc.NavLink("Conclusion", href="#conclusion"),
                                ],
                                vertical=True,
                                pills=True,
                                className="sidebar-nav",
                            ),
                        ],
                        className="sidebar",
                    ),

                    # Main Content
                    html.Div(
                        [
                            html.H1(
                                "Case Study: An Analysis of Data Science Graduates' Salaries",
                                className="title",
                            ),
                            html.H4(
                                "By Neel Deshmukh, Ceferino Malabed, and Arushi Sharma",
                                className="subtitle",
                            ),
                            html.Hr(),
                            html.Div(
                                [
                                    html.H2("Introduction", id="introduction"),
                                    dcc.Markdown(
                                        """
                                        fire intro.
                                        """
                                    ),
                                ],
                                className="section",
                            ),
                            html.Div(
                                [
                                    html.H2("Let's Dive into Linear Regression\n", id="main"),
                                    dcc.Markdown(
                                        """
                                     
                                        ### What can one predictor tell us about data science salaries?

                                        predicting stuff
                                        """
                                    ), 
                                    dcc.Graph(figure=slr_fig),

                                    dcc.Markdown(
                                        """

                                        ### How about multiple predictors?

                                        predicting more stuff

                                        ### Modeling our Data with Random Effects
                                        cool interactive stuff
                                        """
                                    ),
                                    dcc.Graph(figure=mlr_fig)
                                ],
                                className="section",
                            ),
                            html.Div(
                                [
                                    html.H2("Conclusion", id= "conclusion"),
                                    dcc.Markdown(
                                        """
                                        Summarize your findings, thoughts, or next steps.
                                        """
                                    ),
                                ],
                                className="section",
                            ),
                        ],
                        className="content",
                    ),
                ],
            ),
        ],
    )


app.layout = serve_layout

server = app.server

if __name__ == "__main__":
    app.run(debug=True)
