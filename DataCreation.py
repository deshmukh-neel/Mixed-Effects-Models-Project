import numpy as np
import pandas as pd

# Reproducible RNG
rng = np.random.default_rng(42)

# --- config ---
universities = ["UC Berkeley", "Stanford", "UCLA", "UC San Diego", "San Jose State"]
n_per_uni = 100                           # exactly 100 per university

# Per-university GPA targets (more separated means)
gpa_targets = {
    "UC Berkeley": 3.60,
    "Stanford":    3.75,
    "UCLA":        3.55,
    "UC San Diego":3.40,
    "San Jose State": 3.35,
}
gpa_between_sd = 0.07   # jitter around each target (between-school)
gpa_within_sd  = 0.25   # individual variation (within-school)

# Work experience targets (approximate, not perfect) + jitter
work_exp_targets = {
    "UC Berkeley": 3.0,
    "Stanford":    5.0,
    "UCLA":        4.0,
    "UC San Diego":2.0,
    "San Jose State": 6.0,
}
exp_between_sd = 0.4    # per-school mean jitter
exp_within_sd  = 1.5    # per-student spread

# Salary generator knobs
uni_intercept_sd = 15_000
salary_noise_sd  = 15_000

rows = []
for u in universities:
    N = n_per_uni

    # Per-university parameters (jittered)
    mu_gpa_u = np.clip(rng.normal(gpa_targets[u], gpa_between_sd), 2.5, 4.0)
    mu_exp_u = np.clip(rng.normal(work_exp_targets[u], exp_between_sd), 0, 20)

    # Predictors
    gpa = np.clip(rng.normal(mu_gpa_u, gpa_within_sd, N), 2.5, 4.0)
    work_exp = np.clip(rng.normal(mu_exp_u, exp_within_sd, N), 0, 20).round()

    # Skills (must be <= work_exp)
    yrs_py  = np.array([rng.integers(0, int(w) + 1) for w in work_exp])
    yrs_sql = np.array([rng.integers(0, int(w) + 1) for w in work_exp])

    # Salary: signal + uni random intercept + noise, bounded [85k, 300k]
    uni_eff = rng.normal(0, uni_intercept_sd)
    signal = (110_000
              + 35_000*(gpa - 3.0)
              + 10_000*np.log1p(work_exp)
              + 1_500*yrs_py
              + 1_000*yrs_sql
              + uni_eff)
    salary = np.clip(signal + rng.normal(0, salary_noise_sd, N), 85_000, 300_000).round().astype(int)

    rows.append(pd.DataFrame({
        "masters_university": u,
        "masters_gpa": np.round(gpa, 2),
        "relevant_work_years": work_exp.astype(int),
        "years_python": yrs_py.astype(int),
        "years_sql": yrs_sql.astype(int),
        "first_job_salary": salary
    }))

df = pd.concat(rows, ignore_index=True)

# Sanity checks
assert (df["masters_gpa"].between(2.5, 4.0)).all()
assert (df["relevant_work_years"].between(0, 20)).all()
assert ((df["years_python"] <= df["relevant_work_years"]) & (df["years_python"].between(0, 20))).all()
assert ((df["years_sql"]    <= df["relevant_work_years"]) & (df["years_sql"].between(0, 20))).all()
assert (df["first_job_salary"].between(85_000, 300_000)).all()

# Save + quick peek at university means
df.to_csv("masters_salary.csv", index=False)
print("Saved: masters_salary.csv")
print(df.groupby("masters_university")[["masters_gpa","relevant_work_years","years_python","years_sql","first_job_salary"]].mean().round(2))
