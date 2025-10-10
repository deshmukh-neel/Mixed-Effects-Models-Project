import numpy as np
import pandas as pd
import plotly.graph_objs as go



def ols_fit(x, y):
    X = np.c_[np.ones(len(x)), x]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def graph_slr(data_file):
    

    df = pd.read_csv(data_file)
    df.columns = [c.strip() for c in df.columns]


    group_col = "masters_university"
    y_col = "first_job_salary"
    predictors = [
        ("masters_gpa", "Master's GPA"),
        ("relevant_work_years", "Relevant Work Years"),
        ("years_python", "Years Python"),
        ("years_sql", "Years SQL"),
    ]

    colors = {
        "UC Berkeley": "#FDB515",
        "Stanford": "#d62728",
        "UC San Diego": "#00629B",
        "San Jose State": "#7ee081",
        "UCLA": "#bf94e4"
    }


    df = df[[group_col, y_col] + [p[0] for p in predictors]].dropna().copy()
    groups = sorted(df[group_col].unique())


    all_traces = []
    blocks = []  

    for p_idx, (x_col, x_label) in enumerate(predictors):
        dfp = df[[group_col, x_col, y_col]].copy()
        fe_intercept, fe_slope = ols_fit(dfp[x_col].values, dfp[y_col].values)
        x_min, x_max = dfp[x_col].min(), dfp[x_col].max()
        x_line = np.linspace(x_min, x_max, 100)
        y_fe = fe_intercept + fe_slope * x_line

        start_idx = len(all_traces)
        n_point_traces = 0
        n_group_line_traces = 0


        for g in groups:
            dfg = dfp[dfp[group_col] == g]
            color = colors.get(g, "#999999")
            all_traces.append(go.Scatter(
                x=dfg[x_col],
                y=dfg[y_col],
                mode="markers",
                name=g, 
                legendgroup=g,
                hovertemplate=f"{g}<br>{x_label}: %{{x}}<br>{y_col.replace('_',' ').title()}: %{{y}}<extra></extra>",
                marker=dict(size=7, opacity=0.6, color=color, line=dict(width=0)),
                visible=True if x_col == "masters_gpa" else False,  
                showlegend=(p_idx == 0),
        ))
        n_point_traces += 1


    
        for g in groups:
            dfg = dfp[dfp[group_col] == g]
            color = colors.get(g, "#999999")
            if len(dfg) >= 2:
                gi, gs = ols_fit(dfg[x_col].values, dfg[y_col].values)
            else:
                gi = fe_intercept + (dfg[y_col].mean() - (fe_intercept + fe_slope * dfg[x_col].mean()))
                gs = fe_slope
    
            all_traces.append(go.Scatter(
                x=x_line,
                y=gi + gs * x_line,
                mode="lines",
                name=f"{g} line", 
                legendgroup=g,
                hoverinfo="skip",
                line=dict(width=2, color=color),
                visible=True if x_col == "masters_gpa" else "legendonly" if p_idx == 0 else False,
                showlegend=(p_idx == 0),
        ))
        n_group_line_traces += 1

    
        all_traces.append(go.Scatter(
            x=x_line,
            y=y_fe,
            mode="lines",
            name="Population average",
            hovertemplate=f"Fixed: y = {fe_intercept:.2f} + {fe_slope:.2f}×{x_label}<extra></extra>",
            line=dict(width=4, dash="dash", color="black"),
            visible=True if p_idx == 0 else False,
            showlegend=(p_idx == 0),
        ))

        blocks.append((start_idx, n_point_traces, n_group_line_traces, len(all_traces) - 1))

    fig = go.Figure(data=all_traces)

    fig = go.Figure(data=all_traces)

    fig.update_layout(
        title=dict(
            text="An Example of Simple Linear Regression",
            x=.05
        ),
        template="plotly_white",
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9",
        hovermode="closest",
        margin=dict(t=110, r=30, b=90, l=60),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.28,
            xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        yaxis=dict(
            title=y_col.replace("_", " ").title(),
            gridcolor="rgba(0,0,0,0.08)", zeroline=False
        ),
    )
    fig.update_xaxes(title=predictors[0][1])

    # --- Add Radio Buttons to Switch Predictors ---
    buttons = []
    n_groups = len(groups)

    for p_idx, (x_col, x_label) in enumerate(predictors):
        vis = []
        showleg = []
        for idx_block, (start_idx, n_pts, n_lines, fixed_idx) in enumerate(blocks):
            if idx_block == p_idx:
                # ✅ Only Master's GPA shows scatter + lines
                if x_col == "masters_gpa":
                    vis.extend([True] * n_pts)     # show scatter
                    showleg.extend([True] * n_pts)
                else:
                    vis.extend([False] * n_pts)    # hide scatter for others
                    showleg.extend([False] * n_pts)
                vis.extend([True] * n_lines)       # show regression lines
                showleg.extend([True] * n_lines)
                vis.append(True)                   # fixed global line
                showleg.append(True)
            else:
                vis.extend([False] * n_pts)
                showleg.extend([False] * n_pts)
                vis.extend([False] * n_lines)
                showleg.extend([False] * n_lines)
                vis.append(False)
                showleg.append(False)

        buttons.append(dict(
            label=x_label,
            method="update",
            args=[
                {"visible": vis, "showlegend": showleg},
                {"xaxis": {"title": x_label}, "transition": {"duration": 500, "easing": "cubic-in-out"}},
            ],
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.8, xanchor="center",
            y=1.18, yanchor="top",
            showactive=True,
            buttons=buttons,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        )]
    )
    fig.update_layout(width=1200, height=700)

    return fig
