import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.formula.api as smf

def graphs_full(data_file):

    # Load dataset
    df = pd.read_csv(data_file)  

    # Fit a mixed-effects model
    model = smf.mixedlm(
        "first_job_salary ~ masters_gpa + relevant_work_years + years_python + years_sql",
        df,
        groups=df["masters_university"]
    )
    result = model.fit()

    # Add predicted values
    df["predicted_salary"] = result.fittedvalues

    # Get list of universities
    universities = sorted(df["masters_university"].unique())

    # Custom colors
    colors = {
        "UC Berkeley": "#FDB515",    
        "Stanford": "#d62728",       
        "UC San Diego": "#00629B",   
        "San Jose State": "#7ee081", 
        "UCLA": "#bf94e4"            
    }


    # --- Create base figure with all universities ---
    fig = go.Figure()

    for uni in universities:
        group = df[df["masters_university"] == uni]
        coeffs = np.polyfit(group["predicted_salary"], group["first_job_salary"], 1)
        x_line = np.linspace(group["predicted_salary"].min(), group["predicted_salary"].max(), 100)
        y_line = coeffs[0] * x_line + coeffs[1]

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=group["predicted_salary"],
                y=group["first_job_salary"],
                mode="markers",
                name=uni,
                marker=dict(size=7, opacity=0.7, color=colors[uni]),
                hovertext=[uni]*len(group),
                hovertemplate="<b>%{hovertext}</b><br>Predicted: %{x:.0f}<br>Actual: %{y:.0f}<extra></extra>"
            )
        )

        # Regression line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"{uni} Line",
                line=dict(color=colors[uni], width=2),
                showlegend=False
            )
        )

    # --- Define frames for animation ---
    frames = []

    # Frame: "All Universities"
    frame_all = []
    for uni in universities:
        group = df[df["masters_university"] == uni]
        coeffs = np.polyfit(group["predicted_salary"], group["first_job_salary"], 1)
        x_line = np.linspace(group["predicted_salary"].min(), group["predicted_salary"].max(), 100)
        y_line = coeffs[0] * x_line + coeffs[1]

        frame_all.append(
            go.Scatter(
                x=group["predicted_salary"],
                y=group["first_job_salary"],
                mode="markers",
                marker=dict(size=7, opacity=0.7, color=colors[uni]),
                name=uni
            )
        )
        frame_all.append(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=colors[uni], width=2),
                opacity=1,
                showlegend=False
            )
        )

    frames.append(go.Frame(name="All", data=frame_all))

    # Frames for each individual university
    for uni in universities:
        frame_uni = []
        for other_uni in universities:
            group = df[df["masters_university"] == other_uni]
            coeffs = np.polyfit(group["predicted_salary"], group["first_job_salary"], 1)
            x_line = np.linspace(group["predicted_salary"].min(), group["predicted_salary"].max(), 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            opacity = 1 if other_uni == uni else 0.1

            frame_uni.append(
                go.Scatter(
                    x=group["predicted_salary"],
                    y=group["first_job_salary"],
                    mode="markers",
                    marker=dict(size=7, opacity=opacity, color=colors[other_uni]),
                    name=other_uni
                )
            )
            frame_uni.append(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color=colors[other_uni], width=2),
                    opacity=opacity,
                    showlegend=False
                )
            )
        frames.append(go.Frame(name=uni, data=frame_uni))

    # Attach frames
    fig.frames = frames

    # --- Dropdown buttons ---
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

    # --- Layout ---
    fig.update_layout(
        title="Mixed Effects Model: Predicted vs Actual Salary by University",
        xaxis_title="Predicted Salary",
        yaxis_title="Actual Salary",
        template="plotly_white",
        width=1200,
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
