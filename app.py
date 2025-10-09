
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# Example figure
df = px.data.iris()   # built-in dataset
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 title="Iris Sepal Width vs Length")

# Choose a Bootstrap theme (optional) – e.g. BOOTSTRAP, DARKLY, etc.
# But you can also just use CSS custom properties as above.
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
                                    dcc.Graph(figure=fig),

                                    dcc.Markdown(
                                        """

                                        ### How about multiple predictors?

                                        predicting more stuff

                                        ### Modeling our Data with Random Effects
                                        cool interactive stuff
                                        """
                                    ),
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
