
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
                                        If you are reading this, you’re probably applying to be a graduate student, yearning to enter the Data Science field, and wondering what it takes to actually land that first job. Is it GPA? The number of internships you’ve had? The university you attend? The factors are endless. 

                                        
                                        That’s exactly what we wanted to explore – but instead of relying on a polished, public dataset, we decided to build our own from scratch. Why not! We simulated realistic data for recent master’s graduates from five California universities, each with different experience levels, GPAs, and technical skills. As University of San Francisco students, we chose to not include the university. Ignorance is bliss! (Okay, yes, it’s simulated data. But let’s be honest – it feels real enough to hurt.)

                                        Our goal wasn’t just to predict salaries – it was to understand how these factors interact and contribute to the bigger picture. We want to explore this question step by step – starting with simple linear regression  to understand each variable's relationship with salary alone. Moving forward with multiple linear regression to look at the relationship of the factors and salary as a whole,  and introducing mixed-effects models – our main topic for today!

                                        """,
                                        style={
                                        "fontSize": "18px",  
                                        "lineHeight": "1.6",  
                                    }
                                    ),
                                    html.H2("Data", id="data"),
                                    dcc.Markdown(
                                    '''
                        
                                    Each variable was generated within realistic ranges to resemble actual post-graduate data science job offers. For example, the GPAs range from 2.5 to 4.0 because girl, let’s be real, no one can graduate with a 1.8 GPA. 

                                    The variables we included in our simulated dataset as predictors are as follows:
                                    - Masters GPA: on a 4.0 scale
                                    - Relevant Work Experience: total years of relevant professional experience
                                    - Years of Python: how long (in years) student has experience with Python
                                    - Years of SQL: how long (in years) student has experience with SQL
                                    - University: UC Berkeley, Stanford, UCLA, UC San Diego, San Jose State
                                    - First Job Salary: our outcome variable!
                                    ''',
                                    style={
                                        "fontSize": "18px",  
                                        "lineHeight": "1.6",  
                                    }
                                    )
                                ],
                                className="section",

                            ),
                            html.Div(
                                [
                                    html.H2("Let's Dive into Linear Regression\n", id="main"),
                                    dcc.Markdown(
                                        """
                                     
                                        ### Our lesson for today

                                        Like Doja Cat once said, let's “get into it yuh.” 

                                        We started simple - a Simple Linear Regression (SLR) model to see how each variable alone relates to salary. 
                                        Think of it as testing the waters before diving into deeper modeling. 

                                        """,
                                        style={
                                        "fontSize": "18px",  
                                        "lineHeight": "1.6",  
                                    }
                                    ), 
                                    dcc.Graph(figure=slr_fig),
                                    dcc.Markdown(
                                        '''
                                        ### How about multiple predictors?
                                        If only salary was determined by one factor… But we live in a cruel world. This is where Multiple Regression (MLR) comes in. It allows us to look at how the variables collectively influence salary. Our question changes from “Does GPA matter?" to "How much does GPA matter when experience, Python, SQL, and University are also considered? This makes our model more realistic. 

                                        Model: Salary ~ GPA + Work Experience + Python Years + SQL Years + University

                                            ''',
                                        style={
                                            "fontSize": "18px",  
                                            "lineHeight": "1.6",  
                                        }
                                    ),

                                    html.H2("Model Results"),
                                    html.Img(
                                        src="/assets/OLS_Summary.png", 
                                        style={
                                            "width": "70%", 
                                            "display": "block",
                                            "margin": "20px auto", 
                                            "border": "1px solid #ddd", 
                                            "borderRadius": "8px"
                                        }
                                    ),
                                    dcc.Markdown(
                                        """
                                Looking at the summary, our model explains about 57% of the variation in salary – not bad for simulated data.
                                Almost all predictors appear to be statistically significant here, and all have positive relationships with salary. 
                                In plain terms, higher GPA, more experience, more years of programming experience, all tend to correspond to higher pay.
                                  SQL experience though… It seems like SQL didn’t make the cut. If only this dataset were real huh…

                                The university effects are also quite interesting. Compared to San Jose State (our baseline), all other universities show higher predicted salaries, especially San Diego.
                                These differences capture what we call “fixed effects.”

                                Predicted Salary = -32,810 + 39,410.6*(GPA) + 6,868*(Work Experience) + 1,921*(Years Python) + 627*(Years SQL)  + 17,880*(Stanford) + 25,850*(UC Berkeley) + 50,370*(UCSD) + 26,390*(UCLA)

                                So far, our model assumes each university’s effect on salary is fixed and exact. In other words, 
                                it treats the five schools in our dataset as the only ones that exist and assumes that their differences are perfectly estimated. 
                                However, we are pretty aware of the fact that other universities exist with their own graduate programs.
                                Even if this data was real, what if we wanted to generalize beyond these five universities? 
                                        """,
                                        style={
                                        "fontSize": "18px",  
                                        "lineHeight": "1.6",  
                                    }
                                    ),
                                    dcc.Graph(figure=mlr_fig)
                                ],
                                className="section",
                            ),

                            html.Div(
                                [
                                    html.H2("Mixed Effect Models", id= "mixed_effect"),
                                    dcc.Markdown(
                                        '''

                                        From our multiple linear regression model, we’ve seen that universities differ in their average salaries, but how can we represent that in the model? 
                                        A regular multiple linear regression model assumes that the differences between universities are fixed and known. But, in this case, students from the same university share a lot of the same background which means that the differences vary and are unknown. 
                                        Students from the same university might have the same professors, career outlooks through school fairs, or program reputations that affect their salaries.
                                        If we ignore this and use standard multiple linear regression (MLR), we’re pretending those clusters do not exist (*Alexa play Bad by Michael Jackson*).
                                        Mixed-effect models recognize that observations within the same group (in this case, university) are more alike than those from different groups. 
                                        Treating every data point as independent can lead to misleadingly narrow confidence intervals and inflated significance. Essentially, our investigation will be incorrect.

                                        ## What Mixed-Effect Models Do
                                        Mixed-effects models allow us to model both the individual-level effects (like GPA, experience, and skills) and group-level effects 
                                        (like the university someone attended). In our case, we know salaries tend to cluster by university; some schools might consistently 
                                        have graduates who earn more, even after accounting for other factors like GPA. A mixed-effects model captures this by giving each university 
                                        its own random intercept. This means that we are allowing each school to vary slightly around the average rather than assuming the university differences 
                                        are fixed and exact. We are able to recognize that not all variation is equal! Hooray!
                                        ''',
                                        style={
                                            "fontSize": "18px",  
                                            "lineHeight":"1.6",  
                                        }
                                    ),
                                ],
                                className="section"
                            ), 
                            html.Div(
                                [
                                    dcc.Graph(figure=me_fig),
                                    html.H2("Let's break it down."),
                                    dcc.Graph(figure=me_pred_fig)
                                ],
                                className="section"
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
