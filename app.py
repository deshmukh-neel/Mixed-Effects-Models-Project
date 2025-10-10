
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import statsmodels.formula.api as smf
from graphs import *
import pandas as pd
from Plots.graphs_full import *
from Plots.graphs_slr import *


salary_data = pd.read_csv("Data/masters_salary.csv")


me_fig = build_mixed_effects_figure()
me_pred_fig = build_predicted_vs_actual_figure(salary_data)

slr_fig = graph_slr("Data/masters_salary.csv")
mlr_fig = graphs_full("Data/masters_salary.csv")

code_snippet = """```
                model1 = smf.mixedlm("first_job_salary ~ masters_gpa",
                    data=salary_data,
                    groups=salary_data["masters_university"],
                    re_formula="~masters_gpa"
                ).fit()```
                """

equation = r'''
        $$
        \text{Predicted Salary} = -32{,}810 \\
        \; +\; 39{,}410.6 \times \text{GPA} \\
        \; +\; 6{,}868 \times \text{Work Experience} \\
        \; +\; 1{,}921 \times \text{Years Python} \\
        \; +\; 627 \times \text{Years SQL} \\
        \\
        \; +\; 17{,}880 \times \text{Stanford} \\
        \; +\; 25{,}850 \times \text{UC Berkeley} \\
        \; +\; 50{,}370 \times \text{UCSD} \\
        \; +\; 26{,}390 \times \text{UCLA}
        $$
        '''
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def serve_layout():
    return html.Div(
        id="page-container",
        children=[
            dcc.Location(id='url'),
            dbc.Navbar(
                [
                    html.Div("Mixed Effects Models", className="navbar-brand title", 
                             style={
                            "marginLeft": "30px"  # moves it to the right
                }),
                ],
                className="header",
                expand=True,
            ),

            html.Div(
                className="layout-container",
                children=[
                    html.Div(
                        [
                            html.H2("Contents", className="sidebar-title",style={
                            "marginLeft": "18px"
                            }),
                            html.Hr(),
                            dbc.Nav(
                                [
                                    dbc.NavLink("Introduction", href="#introduction", external_link=True),
                                    dbc.NavLink("Simple Linear Regression", href="#slr", external_link=True),
                                    dbc.NavLink("Multiple Linear Regression", href="#mlr", external_link=True),
                                    dbc.NavLink("Mixed Effect Models", href="#mixed_effect", external_link=True),
                                    dbc.NavLink("Conclusion", href="#conclusion", external_link=True),
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
                                        "lineHeight": "1.6"
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
                                    html.H2("Let's Dive into Linear Regression\n", id="slr"),
                                    dcc.Markdown(
                                        """
                                     
                                        ### Our lesson for today

                                        Like Doja Cat once said, let's “get into it yuh.” 

                                        We started simple: a Simple Linear Regression (SLR) model to see how each variable relates to salary on its own. By starting with SLR, we can see which variables show strong linear relationships with salary on their own before forming more complex models. 

                                        The SLR models:
                                        - Salary ~ GPA
                                        - Salary ~ Work Experience (years)
                                        - Salary ~ Internship Experience (years)
                                        - Salary ~ Python Experience (years)
                                        - Salary ~ SQL Experience (years)

                                        """,
                                        style={
                                        "fontSize": "18px",  
                                        "lineHeight": "1.6",  
                                    }
                                    ), 
                                    dcc.Graph(figure=slr_fig),
                                    dcc.Markdown(
                                        '''
                                        In the interactive graph above you can change the graph to reflect how each variable affects the predicted salary in an SLR model.
                                        Toggle the graph with the buttons in the upper right hand corner of the figure to view the effects of each variable: GPA, Work Experience, Python Experience, and SQL Experience.
                                        Each color for the plot points and regression lines are representative for their respective schools. Although there are some variables that don’t have too much of an effect depending on the school,
                                        we can see that each predictor generally has a positive influence on the respective regression lines. This data kind of suggests what we already as a collective know, right?
                                        More knowledge and experience leads to more money. 
                                        ''',
                                        style={
                                            "fontSize": "18px",  
                                            "lineHeight": "1.6",  
                                        }
                                    ),
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

                                    html.H2("Multiple Linear Regression", id = "mlr"),
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
                                    Looking at the summary, our model explains about 57% (R^2 = 0.579) of the variation in salary – not bad for simulated data. Almost all predictors seem to be statistically significant here, and all have positive relationships with salary. In more simple terms, higher GPA, more experience,
                                    more years of programming experience, all tend to correspond to higher pay. SQL experience though… It seems like SQL didn’t make the cut. If only this dataset were real huh…
                                    What do all these numbers mean though? If you look at the “coef” column, these numbers tell us the coefficients for each predictor in the full model. 
                                    Among the predictors, GPA stands out with a strong positive coefficient (around 39,000) meaning that each one point increase in GPA is associated with roughly a $39,000 higher starting salary, while holding all other predictors constant. Relevant work experience also shows to have a strong impact on predicted salary, adding about $6,800 per year of experience.
                                    Python experience has a smaller but still significant positive effect (around 1,9000 per year of experience) suggesting that technical skills pay off (modestly). As we mentioned earlier, SQL experience has a small coefficient ($600) 
                                    and a p-value of 0.185, suggesting that in this model it is not a statistically significant predictor of salary. Sorry to all the SQL developers not getting love…
                                    The university effects are also quite interesting. Compared to San Jose State (our baseline), all other universities show higher predicted salaries, especially San Diego.
                                    These differences capture what we call “fixed effects.”
                                    To help you visualize the model with the coefficients, below is the predicted salary equation (scroll L/R to see the whole thing):

                                """,style={
                                        "fontSize": "18px",  
                                        "lineHeight": "1.6",  
                                    }
                                    ),
                                    dcc.Markdown(equation, mathjax=True,style={
                                        "fontSize": "18px",
                                        "overflowX": "auto",
                                        "maxWidth": "100%",
                                        "whiteSpace": "nowrap"
                                        }),
                                    dcc.Graph(figure=mlr_fig),
                                    dcc.Markdown(
                                        """
                                        The figure above displays an MLR model for each university. Use the dropdown menu to see each school’s MLR line and data separately from one another.
                                        Makes it a little nicer to view, yeah?  
                                        With this, you’re able to see how the variables work together to build a regression line that can help predict the salary of a graduate from that particular university.
                                        At a glance, you’re able to look at a student’s actual salary and where they fall on the graph in comparison to the prediction line. 
                                        Each line shows a positive relationship with the predictors, though it seems that San Jose State seems to have a flatter trend compared to the other schools. 
                                        """,
                                        style ={
                                            "fontSize": "18px",  
                                            "lineHeight": "1.6",
                                        }
                                    ),
                                    dcc.Markdown(
                                        """
                                    So far, our model assumes each university’s effect on salary is fixed and exact. In other words, 
                                    it treats the five schools in our dataset as the only ones that exist and assumes that their differences are perfectly estimated. 
                                    How could we recognize that each university might have its own natural variation that we do not want to overfit?
                                    This is exactly where mixed-effects models come in! 
                                        """,
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
                                    html.H2("Mixed Effect Models", id= "mixed_effect"),
                                    dcc.Markdown(
                                        '''

                                        From our multiple linear regression model (MLR), we’ve seen that student's average salaries differ by university, but how can we be sure we modeled it well?
                                        A standard MLR model assumes that the categorical variable, the master's program a student attended, is a fixed quantity.  However, as categories
                                        start to have 5 or so levels, it becomes difficult to defend the decision to model it as a fixed variable.  That's where randomness comes in. It's not *mathematical* randomness,
                                        but since we have a group effect we'd like to model (master's programs), we consider the master's program to be a *random* effect.  Intuitively, think about the differences 
                                        between data science programs from 5 different schools. Each school has a different set of professors, classes, specialization topics and reputation.  Some programs can be done remotely, 
                                        and others are in person. For the next step of our analysis, we'd like to model how different master's programs can affect each of our other predictors' impact on salary.
                                        If we ignore the group effect and use standard MLR, we’re pretending those clusters do not exist (*Alexa play Bad by Michael Jackson*).
                                        Mixed-effect models recognize that observations within the same group are more alike than those from different groups. 
                                        Treating every data point as independent can lead to misleadingly narrow confidence intervals and inflated significance. Essentially, our investigation will be incorrect.

                                        ## What Mixed-Effect Models Do
                                        Mixed-effects models allow us to model both the individual-level effects (like GPA, experience, and skills) and group-level effects.
                                        In our case, we know salaries tend to cluster by university; some schools might consistently 
                                        have graduates who earn more, even after accounting for other factors like GPA. Before we move on, there are a few terms we should go over.
                                        Mixed effects models have a few different names. They are also known as hierarchical or multilevel models. 
                                        The name we used to describe our model depends on the structure of our data and how we collect it.  For example, longitudinal studies can also
                                        be modeled with mixed effects. (Think assessing a student's test scores over multiple semesters.) 
                                        Since we want to model the random effects of our 5 universities, we're clustering the students by a university's master's program,
                                        as alluded to previously.  For the purposes of our blog, we'll focus on the two most common concepts associated with mixed effects models:
                                        random slopes and random intercepts. A random intercepts model is the most common and simple example of a mixed effects model.
                                        A mixed-effects model captures this by giving each university 
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
                                    dcc.Graph(figure=me_pred_fig),
                                    dcc.Markdown(code_snippet, style={"backgroundColor": "#f8f9fa", "padding": "1rem", "borderRadius": "6px"})
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
