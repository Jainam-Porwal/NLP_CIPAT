# Resume ↔ Job Matching (NLP)

Predict how well a resume matches a job posting (score 0–1) and use it to **rank** candidates or jobs.
Dataset: `data/resume_data_for_ranking.csv` (9,544 resume↔job pairs = 344 unique resumes × 28 jobs).

## Project structure
```
resume_job_matching/
├── resume_job_matching.ipynb   # EDA → preprocessing → features → models → evaluation → save artifacts
├── utils.py                    # shared cleaning / feature code (used by notebook AND app)
├── app.py                      # Streamlit demo
├── requirements.txt
├── data/resume_data_for_ranking.csv
└── models/                     # created by the notebook (matcher.joblib, jobs.joblib, resumes.joblib, ...)
```

## Run it
```bash
pip install -r requirements.txt

# 1) (re)create the notebook outputs + model artifacts
jupyter notebook resume_job_matching.ipynb      # Run All   (≈ 5–10 min on a laptop, mostly hyper-parameter search)

# 2) launch the demo
streamlit run app.py
```
The `models/` folder is already included, so you can run step 2 straight away.

## Approach in one paragraph
Resume and job fields are parsed/cleaned, then turned into **TF-IDF similarities** (whole text, skills, responsibilities,
title, education), **skill-overlap** and **experience-gap** features, plus **LSA (SVD)** topic vectors. A tuned
**HistGradientBoostingRegressor** predicts the score. Evaluation uses a **split grouped by resume** (the same resume
appears ~28 times, so a random split would leak) and reports R², MAE, per-job Spearman and NDCG@10.

## App features
* Match a resume (typed, or loaded from a sample) with one of the 28 jobs **or your own job description**
* Similarity breakdown, matched / missing required skills, experience check
* Rank all 28 jobs for one resume · rank the 344 dataset resumes for one job
* Model comparison & feature importance
