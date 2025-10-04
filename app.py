
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

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
                                    html.H2("Let's Dive in....\n", id="main"),
                                    dcc.Markdown(
                                        """
                                     
                                        ### Intro to Linear Regression

                                        predicting stuff

                                        ### Expanding to Multiple Linear Regression

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
    app.run_server(debug=True)
