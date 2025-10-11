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

salary_data = pd.read_csv("Data/masters_salary.csv")

model1 = smf.mixedlm(
    "first_job_salary ~ masters_gpa",
    data=salary_data,
    groups=salary_data["masters_university"],
    re_formula="~masters_gpa"
).fit()

model2 = smf.mixedlm(
    "first_job_salary ~ relevant_work_years",
    data=salary_data,
    groups=salary_data["masters_university"],
    re_formula="~relevant_work_years"
).fit()

model3 = smf.mixedlm(
    "first_job_salary ~ years_python",
    data=salary_data,
    groups=salary_data["masters_university"],
    re_formula="~years_python"
).fit()

model4 = smf.mixedlm(
    "first_job_salary ~ years_sql",
    data=salary_data,
    groups=salary_data["masters_university"],
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

def build_mixed_effects_figure():
    traces_gpa   = create_spaghetti_traces(model1, 'masters_gpa', salary_data)
    traces_work  = create_spaghetti_traces(model2, 'relevant_work_years', salary_data)
    traces_py    = create_spaghetti_traces(model3, 'years_python', salary_data)
    traces_sql   = create_spaghetti_traces(model4, 'years_sql', salary_data)

    fig = go.Figure()

    for i, trace in enumerate(traces_gpa + traces_work + traces_py + traces_sql):
        trace.visible = (i < len(traces_gpa))
        fig.add_trace(trace)

    n_gpa = len(traces_gpa)
    n_work = len(traces_work)
    n_py = len(traces_py)
    n_sql = len(traces_sql)
    buttons = [
            dict(
                label='GPA',
                method='update',
                args=[
                    {'visible': [True]*n_gpa + [False]*(n_work+n_py+n_sql)},
                    {
                        'title': {'text': 'Mixed Effects: GPA vs First Job Salary'},
                        'xaxis': {'title': {'text': "GPA"}},
                        'yaxis': {'title': {'text': "First Job Salary"}},
                        'legend': {'title': {'text': "Master's Program University"}}
                    }
                ]
            ),
            dict(
                label='Work Experience',
                method='update',
                args=[
                    {'visible': [False]*n_gpa + [True]*n_work + [False]*(n_py+n_sql)},
                    {
                        'title': {'text': 'Mixed Effects: Relevant Work Experience vs Salary'},
                        'xaxis': {'title': {'text': "Years of Relevant Work Experience"}},
                        'yaxis': {'title': {'text': "First Job Salary"}},
                        'legend': {'title': {'text': "Master's Program University"}}
                    }
                ]
            ),
            dict(
                label='Python Experience',
                method='update',
                args=[
                    {'visible': [False]*(n_gpa+n_work) + [True]*n_py + [False]*n_sql},
                    {
                        'title': {'text': 'Mixed Effects: Python Experience vs Salary'},
                        'xaxis': {'title': {'text': "Years of Python Experience"}},
                        'yaxis': {'title': {'text': "First Job Salary"}},
                        'legend': {'title': {'text': "Master's Program University"}}
                    }
                ]
            ),
            dict(
                label='SQL Experience',
                method='update',
                args=[
                    {'visible': [False]*(n_gpa+n_work+n_py) + [True]*n_sql},
                    {
                        'title': {'text': 'Mixed Effects: SQL Experience vs Salary'},
                        'xaxis': {'title': {'text': "Years of SQL Experience"}},
                        'yaxis': {'title': {'text': "First Job Salary"}},
                        'legend': {'title': {'text': "Master's Program University"}}
                    }
                ]
            ),
        ]

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            x=0.5, y=1.0,
            buttons=buttons,
            showactive=True
        )],
        title={'text': 'Mixed Effects: GPA vs First Job Salary'},
        xaxis={'title': {'text': 'GPA'}},
        yaxis={'title': {'text': 'First Job Salary'}},
        legend={'title': {'text': "Master's Program University"}},
        template='plotly_white',
        height=700,
        width=1300
    )

    return fig

def build_predicted_vs_actual_figure(data: pd.DataFrame):

    colors = {
        "UC Berkeley": "#FDB515",
        "Stanford": "#d62728",
        "UC San Diego": "#00629B",
        "San Jose State": "#7ee081",
        "UCLA": "#bf94e4"
    }

    universities = sorted(data['masters_university'].unique())

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
            marker=dict(size=7, opacity=0.8, color=colors[uni]),
            hovertemplate=f"<b>{uni}</b><br>Predicted: %{{x:.0f}}<br>Actual: %{{y:.0f}}<extra></extra>"
        ))

    frames = []

    all_frame_data = []
    for i, trace in enumerate(fig.data):
        trace_copy = trace.to_plotly_json()
        if i == 0:
            trace_copy['opacity'] = 1
        else:
            trace_copy['opacity'] = 0.8
        all_frame_data.append(trace_copy)
    frames.append(go.Frame(name="All", data=all_frame_data))
    for uni in universities:
        frame_data = []
        for i, trace in enumerate(fig.data):
            trace_copy = trace.to_plotly_json()
            if i == 0:
                trace_copy['opacity'] = 1 
            elif trace['name'] == uni:
                trace_copy['opacity'] = 1
            else:
                trace_copy['opacity'] = 0.1
            frame_data.append(trace_copy)
        frames.append(go.Frame(name=uni, data=frame_data))

    fig.frames = frames
    buttons = [
        dict(
            label="All Universities",
            method="animate",
            args=[
                ["All"],
                {"frame": {"duration": 500, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 400}}
            ]
        )
    ]

    for uni in universities:
        buttons.append(
            dict(
                label=uni,
                method="animate",
                args=[
                    [uni],
                    {"frame": {"duration": 500, "redraw": True},
                     "mode": "immediate",
                     "transition": {"duration": 400}}
                ]
            )
        )

    fig.update_layout(
        title="Mixed Effect Model: Predicted vs Actual Salary by University",
        xaxis_title="Predicted Salary",
        yaxis_title="Actual Salary",
        template="plotly_white",
        width=1300,
        height=700,
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 1.18,
            "xanchor": "left",
            "y": 1.05,
            "yanchor": "top"
        }],
        legend_title_text="University"
    )

    return fig