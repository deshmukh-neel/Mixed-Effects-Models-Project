import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
data = pd.read_csv('../Data/masters_salary.csv')
model = smf.ols('first_job_salary ~ masters_gpa + relevant_work_years + years_python + years_sql + C(masters_university)',data=data).fit()
print(model.summary())
sm.stats.anova_lm(model, typ=2)