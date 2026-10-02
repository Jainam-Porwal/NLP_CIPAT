"""
Resume <-> Job Matching demo.   Run with:  streamlit run app.py
Needs the artifacts created by resume_job_matching.ipynb (folder ./models).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import (ResumeJobMatcher, build_job_fields, match_required_skills,  # noqa: F401  (class needed for joblib)
                   resume_fields_from_input)

MODELS = Path(__file__).parent / "models"
st.set_page_config(page_title="Resume ↔ Job Matcher", page_icon="📄", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    matcher = joblib.load(MODELS / "matcher.joblib")
    jobs = joblib.load(MODELS / "jobs.joblib")
    resumes = joblib.load(MODELS / "resumes.joblib")
    scores = pd.read_csv(MODELS / "scores.csv")
    metrics = json.load(open(MODELS / "metrics.json"))
    return matcher, jobs, resumes, scores, metrics


if not (MODELS / "matcher.joblib").exists():
    st.error("Model artifacts not found. Run all cells of `resume_job_matching.ipynb` first (it creates ./models).")
    st.stop()

matcher, JOBS, RESUMES, SCORES, METRICS = load_artifacts()
JOB_BY_NAME = {j["name"]: j for j in JOBS}
JOB_NAMES = sorted(JOB_BY_NAME)


def verdict(score: float):
    if score >= 0.78:
        return "Strong match", "🟢"
    if score >= 0.62:
        return "Good match", "🟡"
    if score >= 0.48:
        return "Partial match", "🟠"
    return "Weak match", "🔴"


# ----------------------------------------------------------- session state
FORM_KEYS = {"objective": "", "skills": "", "positions": "", "exp_skills": "", "education": "", "certs": "", "exp_years": 0.0}
for k, v in FORM_KEYS.items():
    st.session_state.setdefault(f"f_{k}", v)


def load_sample(sample):
    for k in FORM_KEYS:
        st.session_state[f"f_{k}"] = sample[k] if k != "exp_years" else float(sample[k])


def current_resume():
    return resume_fields_from_input(**{k: st.session_state[f"f_{k}"] for k in FORM_KEYS})


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📄 Resume ↔ Job Matcher")
    st.caption("NLP demo: TF-IDF + LSA + skill-overlap features → gradient boosting regressor.")
    tm = METRICS["test_metrics"]
    st.subheader("Model quality (held-out resumes)")
    c1, c2 = st.columns(2)
    c1.metric("R²", f"{tm['R2']:.2f}")
    c2.metric("MAE", f"{tm['MAE']:.3f}")
    c1.metric("Spearman / job", f"{tm['Spearman/job']:.2f}")
    c2.metric("NDCG@10", f"{tm['NDCG@10']:.2f}")
    st.caption(f"Trained on {METRICS['n_pairs']:,} pairs · {METRICS['n_resumes']} resumes · {METRICS['n_jobs']} jobs. "
               "Metrics come from an 80/20 split grouped by resume.")
    st.divider()
    st.subheader("📥 Load a sample resume")
    sid = st.selectbox("Resume from dataset", [r["id"] for r in RESUMES], format_func=lambda i: f"Resume #{i}")
    if st.button("Load into form"):
        load_sample(next(r for r in RESUMES if r["id"] == sid))
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Match resume to a job", "🔎 Best jobs for this resume",
                                  "🏆 Rank candidates for a job", "📊 Model info"])

# ---------------------------------------------------------------- resume form
with tab1:
    st.subheader("1 · Candidate resume")
    a, b = st.columns(2)
    with a:
        st.text_area("Career objective / summary", key="f_objective", height=110)
        st.text_area("Skills (comma separated)", key="f_skills", height=110)
        st.text_area("Positions held (comma separated)", key="f_positions", height=70)
    with b:
        st.text_area("Skills used in past jobs (comma separated)", key="f_exp_skills", height=110)
        st.text_input("Education (field of study, degree)", key="f_education")
        st.text_input("Certifications (comma separated)", key="f_certs")
        st.number_input("Total years of experience", min_value=0.0, max_value=50.0, step=0.5, key="f_exp_years")

    st.subheader("2 · Job posting")
    mode = st.radio("Job source", ["Pick from the 28 postings in the dataset", "Write my own job description"], horizontal=True)
    if mode.startswith("Pick"):
        job_name = st.selectbox("Job", JOB_NAMES)
        job = JOB_BY_NAME[job_name]
        with st.expander("Show job details"):
            st.markdown(f"**Experience:** {job['experience']}  \n**Education:** {job['education']}")
            st.markdown("**Required skills:**\n\n" + (job["skills_required"].replace("\n", ", ") or "—"))
            st.markdown("**Responsibilities:**\n\n" + job["responsibilities"].replace("\n", " · "))
        job_fields = job["fields"]
    else:
        j1, j2 = st.columns(2)
        with j1:
            title = st.text_input("Job title", "Machine Learning Engineer")
            resp = st.text_area("Responsibilities (one per line)", "Train and deploy NLP models\nBuild data pipelines\nCollaborate with product teams", height=120)
        with j2:
            req = st.text_area("Required skills (one per line)", "Python\nPyTorch\nSQL\nDocker", height=120)
            edu = st.text_input("Education requirement", "B.Sc in Computer Science")
            yrs = st.number_input("Minimum years of experience", 0, 30, 2)
        job_name = title
        job_fields = build_job_fields({"job_position_name": title, "responsibilities": resp, "skills_required": req,
                                       "educationaL_requirements": edu, "experiencere_requirement": f"At least {yrs} years"})

    if st.button("🚀 Calculate match", type="primary"):
        res = current_resume()
        if not (res["skills_text"] or res["objective_text"] or res["positions_text"]):
            st.warning("Please enter at least some skills, a summary or positions.")
        else:
            score = float(matcher.predict([res], [job_fields])[0])
            feats = matcher.features([res], [job_fields]).iloc[0]
            label, icon = verdict(score)
            matched, missing = match_required_skills(res, job_fields)

            st.divider()
            m1, m2 = st.columns([1, 2])
            with m1:
                st.metric("Predicted match score", f"{score:.0%}")
                st.progress(min(max(score, 0.0), 1.0))
                st.markdown(f"### {icon} {label}")
            with m2:
                st.markdown("**Why? Similarity breakdown (0 – 100 %)**")
                bd = pd.DataFrame({
                    "Signal": ["Whole resume ↔ whole job", "Skills ↔ required skills", "Experience ↔ responsibilities",
                               "Positions ↔ job title", "Education ↔ requirement", "Required-skill coverage"],
                    "Value": [feats["cos_full"], feats["cos_skills"], feats["cos_resp"], feats["cos_title"],
                              feats["cos_edu"], feats["skill_coverage"]]}).set_index("Signal")
                st.bar_chart(bd * 100, horizontal=True, height=250)

            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f"**✅ Required skills found ({len(matched)})**")
                st.write(", ".join(matched) if matched else "—")
            with s2:
                st.markdown(f"**❌ Possible gaps ({len(missing)})**")
                st.write(", ".join(missing) if missing else "—")
            with s3:
                st.markdown("**🕒 Experience**")
                st.write(f"Candidate: **{res['exp_years']:.1f} yrs** · Required: **≥ {job_fields['req_exp_min']:.0f} yrs**")

# --------------------------------------------------------- jobs for a resume
with tab2:
    st.subheader("Which of the 28 jobs fits this resume best?")
    st.caption("Uses the resume currently entered in the form on the first tab.")
    if st.button("Rank all jobs", key="rank_jobs"):
        res = current_resume()
        if not (res["skills_text"] or res["objective_text"] or res["positions_text"]):
            st.warning("Fill in the resume on the first tab (or load a sample from the sidebar).")
        else:
            sc = matcher.predict([res] * len(JOBS), [j["fields"] for j in JOBS])
            out = pd.DataFrame({"Job": [j["name"] for j in JOBS], "Predicted match": sc}).sort_values("Predicted match", ascending=False)
            st.bar_chart(out.set_index("Job")["Predicted match"], horizontal=True, height=620)
            out["Predicted match"] = (out["Predicted match"] * 100).round(1).astype(str) + " %"
            st.dataframe(out.reset_index(drop=True), hide_index=True)

# ------------------------------------------------------ candidates for a job
with tab3:
    st.subheader("Rank the dataset's 344 resumes for one job")
    jn = st.selectbox("Job", JOB_NAMES, key="rank_job")
    topn = st.slider("Show top N", 5, 30, 10)
    if st.button("Rank candidates", key="rank_cands"):
        jf = JOB_BY_NAME[jn]["fields"]
        sc = matcher.predict([r["fields"] for r in RESUMES], [jf] * len(RESUMES))
        tbl = pd.DataFrame({"resume_id": [r["id"] for r in RESUMES], "Predicted": sc,
                            "Skills": [r["skills"][:90] + ("…" if len(r["skills"]) > 90 else "") for r in RESUMES],
                            "Positions": [r["positions"][:60] for r in RESUMES],
                            "Exp (yrs)": [r["exp_years"] for r in RESUMES]})
        actual = SCORES[SCORES.job == jn].set_index("resume_id")["matched_score"]
        tbl["Dataset score"] = tbl["resume_id"].map(actual)
        tbl = tbl.sort_values("Predicted", ascending=False).head(topn).reset_index(drop=True)
        tbl.index += 1
        st.dataframe(tbl.style.format({"Predicted": "{:.2f}", "Dataset score": "{:.2f}", "Exp (yrs)": "{:.1f}"}))
        st.caption("⚠️ The model was fitted on all dataset resumes, so 'Dataset score' agreement here is optimistic; "
                   "see the Model info tab for honest held-out metrics.")

# --------------------------------------------------------------- model info
with tab4:
    st.subheader("Model comparison (held-out resumes)")
    allm = pd.DataFrame(METRICS["all_models"]).set_index("model")
    st.dataframe(allm.style.format("{:.3f}").highlight_max(axis=0, color="#c6efce"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top features (permutation importance)**")
        st.bar_chart(pd.Series(METRICS["top_features"]).sort_values(), horizontal=True)
    with c2:
        st.markdown("**How it works**")
        st.markdown("""
1. Resume & job fields are parsed and **cleaned** (lower-case, tech-token aware, stop-words removed).
2. **TF-IDF (1–2-grams)** gives similarities between resume and job fields; **skill overlap** and **experience gap** are added.
3. **LSA (SVD)** compresses text into 40 latent topics for resume and job.
4. A tuned **HistGradientBoostingRegressor** predicts the 0 – 1 match score.
        """)
        st.json(METRICS["best_params"], expanded=False)
