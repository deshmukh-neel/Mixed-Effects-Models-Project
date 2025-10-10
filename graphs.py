import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Load data ---
data = pd.read_csv("Data/masters_salary.csv")

# --- Fit mixed effects models ---
model1 = smf.mixedlm(
    "first_job_salary ~ masters_gpa",
    data=data,
    groups=data["masters_university"],
    re_formula="~masters_gpa"
).fit()

model2 = smf.mixedlm(
    "first_job_salary ~ relevant_work_years",
    data=data,
    groups=data["masters_university"],
    re_formula="~relevant_work_years"
).fit()

model3 = smf.mixedlm(
    "first_job_salary ~ years_python",
    data=data,
    groups=data["masters_university"],
    re_formula="~years_python"
).fit()

model4 = smf.mixedlm(
    "first_job_salary ~ years_sql",
    data=data,
    groups=data["masters_university"],
    re_formula="~years_sql"
).fit()

def create_spaghetti_traces(model, x_var, data, group_name='masters_university'):
    colors = {
    "UC Berkeley": "#FDB515",
    "Stanford": "#D62728",
    "UC San Diego": "#00629B",
    "San Jose State": "#7EE081",
    "UCLA": "#BF94E4"
            }
    x_vals = np.linspace(data[x_var].min(), data[x_var].max(), 100)
    traces = []
    fixed_intercept = model.fe_params['Intercept']
    fixed_slope = model.fe_params[x_var]
    y_fixed = fixed_intercept + fixed_slope * x_vals

    traces.append(go.Scatter(
        x=x_vals,
        y=y_fixed,
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Population Average'
    ))

    # Random effects (group-specific lines)
    for group, effects in model.random_effects.items():
        group_intercept = fixed_intercept + effects['Group']
        group_slope = fixed_slope + effects[x_var]
        y_group = group_intercept + group_slope * x_vals

        traces.append(go.Scatter(
            x=x_vals,
            y=y_group,
            mode='lines',
            name=str(group),
            line=dict(color=colors.get(group, '#999999')), 
            opacity=0.7
        ))
    return traces

# --- Build the figure ---
def build_mixed_effects_figure():
    traces_gpa   = create_spaghetti_traces(model1, 'masters_gpa', data)
    traces_work  = create_spaghetti_traces(model2, 'relevant_work_years', data)
    traces_py    = create_spaghetti_traces(model3, 'years_python', data)
    traces_sql   = create_spaghetti_traces(model4, 'years_sql', data)

    fig = go.Figure()

    # Add traces with GPA visible initially
    for i, trace in enumerate(traces_gpa + traces_work + traces_py + traces_sql):
        trace.visible = (i < len(traces_gpa))
        fig.add_trace(trace)

    n_gpa = len(traces_gpa)
    n_work = len(traces_work)
    n_py = len(traces_py)
    n_sql = len(traces_sql)

    buttons = [
        dict(label='GPA',
             method='update',
             args=[{'visible': [True]*n_gpa + [False]*(n_work+n_py+n_sql)},
                   {'title': 'Mixed Effects: GPA vs First Job Salary',
                    'xaxis': {'title': 'GPA'}}]),
        dict(label='Work Experience',
             method='update',
             args=[{'visible': [False]*n_gpa + [True]*n_work + [False]*(n_py+n_sql)},
                   {'title': 'Mixed Effects: Relevant Work Experience vs Salary',
                    'xaxis': {'title': 'Years of Relevant Work Experience'}}]),
        dict(label='Python Experience',
             method='update',
             args=[{'visible': [False]*(n_gpa+n_work) + [True]*n_py + [False]*n_sql},
                   {'title': 'Mixed Effects: Python Experience vs Salary',
                    'xaxis': {'title': 'Years of Python Experience'}}]),
        dict(label='SQL Experience',
             method='update',
             args=[{'visible': [False]*(n_gpa+n_work+n_py) + [True]*n_sql},
                   {'title': 'Mixed Effects: SQL Experience vs Salary',
                    'xaxis': {'title': 'Years of SQL Experience'}}])
    ]

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            x=0.5,
            y=1.0,
            buttons=buttons,
            showactive=True
        )],
        title='Mixed Effects: GPA vs First Job Salary',
        xaxis_title='GPA',
        yaxis_title='First Job Salary',
        legend_title="Master's Program University",
        template='plotly_white',
        height=700
    )

    return fig

def build_predicted_vs_actual_figure(data: pd.DataFrame):
    universities = data['masters_university'].unique()
    color_map = {uni: color for uni, color in zip(universities, px.colors.qualitative.Plotly)}
    model_full = smf.mixedlm(
        "first_job_salary ~ masters_gpa + relevant_work_years + years_python + years_sql",
        data=data,
        groups=data["masters_university"],
        re_formula="~masters_gpa + relevant_work_years + years_python + years_sql"
    ).fit()

    data = data.copy()
    data['pred_full'] = model_full.predict()

    min_val = min(data['first_job_salary'].min(), data['pred_full'].min())
    max_val = max(data['first_job_salary'].max(), data['pred_full'].max())

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Perfect Prediction'
    ))

    for uni in universities:
        subset = data[data['masters_university'] == uni]
        fig.add_trace(go.Scatter(
            x=subset['pred_full'],
            y=subset['first_job_salary'],
            mode='markers',
            name=uni,
            marker=dict(
                size=7,
                opacity=0.7,
                color=color_map[uni]
            )
        ))

    fig.update_layout(
        title=f'Predicted vs Actual: Full Mixed Effects Model',
        xaxis_title='Predicted Salary',
        yaxis_title='Actual Salary',
        template='plotly_white',
        width=1400,      
        height=700,
        legend_title="Master's Program University"
    )

    return fig