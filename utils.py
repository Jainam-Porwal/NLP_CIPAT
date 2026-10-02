"""
Shared helpers for the Resume <-> Job matching project.
Used by BOTH the notebook (training) and the Streamlit app (inference),
so that preprocessing is identical in both places.
"""
import ast
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------------ columns
JOB_COLS = [
    "job_position_name", "responsibilities", "responsibilities.1",
    "educationaL_requirements", "experiencere_requirement",
    "age_requirement", "skills_required",
]
TARGET = "matched_score"

# ------------------------------------------------------------ text cleaning
_STOP = set(
    "a an the and or of in on for to with at by from as is are was were be been this that "
    "these those it its into using use used etc will can able good strong knowledge "
    "experience work working skills skill ability".split()
)
_JUNK = {"n/a", "na", "none", "nan", "null", ""}


def clean_text(text: str) -> str:
    """lowercase, keep letters/digits and a few tech symbols (c++, c#, .net), collapse spaces"""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"c\+\+", " cplusplus ", t)
    t = re.sub(r"c#", " csharp ", t)
    t = re.sub(r"\.net", " dotnet ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    toks = [w for w in t.split() if w not in _STOP and len(w) > 1]
    return " ".join(toks)


def parse_list(x):
    """Columns such as skills are stored as *strings* that look like python lists."""
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else [v]
    except Exception:
        return [s.strip(" '\"") for s in x.strip("[]").split(",")]


def flatten(lst):
    out = []
    for i in lst:
        if isinstance(i, list):
            out.extend(flatten(i))
        elif isinstance(i, str) and i.strip().lower() not in _JUNK:
            out.append(i.strip())
    return out


def norm_skill(s: str) -> str:
    return clean_text(s)


# --------------------------------------------------------- experience years
def _year(s):
    m = re.search(r"(19|20)\d{2}", str(s))
    return int(m.group()) if m else None


def resume_experience_years(starts, ends) -> float:
    """Sum of (end-start) over all jobs. 'Till Date' / 'Present' => current year."""
    now = datetime.now().year
    total = 0.0
    for s, e in zip(parse_list(starts), parse_list(ends)):
        ys = _year(s)
        ye = _year(e)
        if ye is None and isinstance(e, str) and re.search(r"till|present|current|now", e, re.I):
            ye = now
        if ys and ye and ye >= ys:
            total += min(ye - ys, 40)
    return float(min(total, 50))


def required_experience(text):
    """'At least 5 years' -> (5, 5) ; '1 to 3 years' -> (1, 3); missing -> (0, 0)"""
    if not isinstance(text, str):
        return 0.0, 0.0
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if not nums:
        return 0.0, 0.0
    return float(nums[0]), float(nums[-1])


# ------------------------------------------------------ build text per side
def _assemble_resume_fields(skills, positions, job_skills, edu_terms, cert, objective, exp_years) -> dict:
    return {
        "skills_list": sorted({norm_skill(s) for s in list(skills) + list(job_skills) if norm_skill(s)}),
        "skills_text": clean_text(" ".join(skills)),
        "positions_text": clean_text(" ".join(positions)),
        "exp_skills_text": clean_text(" ".join(job_skills)),
        "edu_text": clean_text(" ".join(edu_terms)),
        "cert_text": clean_text(" ".join(cert)),
        "objective_text": clean_text(objective or ""),
        "n_positions": len(positions),
        "n_certs": len(cert),
        "exp_years": float(exp_years),
    }


def build_resume_fields(row) -> dict:
    """Turn one raw dataframe row (or dict) into cleaned resume text fields."""
    objective = row.get("career_objective")
    return _assemble_resume_fields(
        skills=flatten(parse_list(row.get("skills"))),
        positions=flatten(parse_list(row.get("positions"))),
        job_skills=flatten(parse_list(row.get("related_skils_in_job"))),
        edu_terms=flatten(parse_list(row.get("major_field_of_studies"))) + flatten(parse_list(row.get("degree_names"))),
        cert=flatten(parse_list(row.get("certification_skills"))),
        objective=objective if isinstance(objective, str) else "",
        exp_years=resume_experience_years(row.get("start_dates"), row.get("end_dates")),
    )


def _split_items(x):
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return [i.strip() for i in re.split(r"[,\n;]", x or "") if i.strip()]


def resume_fields_from_input(objective="", skills="", positions="", exp_skills="",
                             education="", certs="", exp_years=0.0) -> dict:
    """Same output as build_resume_fields, but from free-text form inputs (used by the Streamlit app)."""
    return _assemble_resume_fields(_split_items(skills), _split_items(positions), _split_items(exp_skills),
                                   _split_items(education), _split_items(certs), objective, exp_years)


def resume_full_text(f: dict) -> str:
    return " ".join([f["objective_text"], f["skills_text"], f["positions_text"],
                     f["exp_skills_text"], f["edu_text"], f["cert_text"]]).strip()


def build_job_fields(row) -> dict:
    req_lines = [l.strip(" •\t") for l in str(row.get("skills_required") or "").split("\n")]
    req_lines = [l for l in req_lines if l and l.lower() not in _JUNK]
    resp = row.get("responsibilities") if isinstance(row.get("responsibilities"), str) else ""
    title = row.get("job_position_name") if isinstance(row.get("job_position_name"), str) else ""
    edu = row.get("educationaL_requirements") if isinstance(row.get("educationaL_requirements"), str) else ""
    lo, hi = required_experience(row.get("experiencere_requirement"))
    return {
        "title_raw": title.strip(),
        "req_skills_raw": req_lines,
        "title_text": clean_text(title),
        "resp_text": clean_text(resp),
        "req_skills_list": sorted({norm_skill(s) for s in req_lines if norm_skill(s)}),
        "req_skills_text": clean_text(" ".join(req_lines)),
        "edu_req_text": clean_text(edu),
        "req_exp_min": lo,
        "req_exp_max": hi,
    }


def job_full_text(j: dict) -> str:
    return " ".join([j["title_text"], j["resp_text"], j["req_skills_text"], j["edu_req_text"]]).strip()


# --------------------------------------------------------- pair features
def _tok_set(texts):
    return {w for t in texts for w in t.split()}


def skill_overlap(resume_skills, req_skills):
    """token-level overlap between resume skills and required skills"""
    r = _tok_set(resume_skills)
    q = _tok_set(req_skills)
    if not q or not r:
        return 0.0, 0.0, 0.0
    inter = len(r & q)
    return inter / len(q), inter / len(r), inter / len(r | q)  # coverage, precision, jaccard


def cos(vec, a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    m = vec.transform([a, b])
    return float(m[0].multiply(m[1]).sum())  # rows are L2-normalised


FEATURE_NAMES = [
    "cos_full", "cos_skills", "cos_resp", "cos_title", "cos_objective", "cos_edu",
    "skill_coverage", "skill_precision", "skill_jaccard", "n_matched_skill_tokens",
    "n_skills", "resume_words", "n_positions", "n_certs", "has_objective",
    "exp_years", "req_exp_min", "req_exp_max", "exp_gap", "n_req_skills", "job_words",
]


def pair_features_batch(vec, R, J):
    """
    R: list of resume-field dicts, J: list of job-field dicts (same length, paired by index).
    vec: fitted TfidfVectorizer (L2-normalised output).
    Returns DataFrame[FEATURE_NAMES]
    """
    n = len(R)
    rf = [resume_full_text(r) for r in R]
    jf = [job_full_text(j) for j in J]

    def pair_cos(a_list, b_list):
        A = vec.transform(a_list)
        B = vec.transform(b_list)
        return np.asarray(A.multiply(B).sum(axis=1)).ravel()

    out = pd.DataFrame(index=range(n))
    out["cos_full"] = pair_cos(rf, jf)
    out["cos_skills"] = pair_cos([r["skills_text"] + " " + r["exp_skills_text"] for r in R],
                                 [j["req_skills_text"] for j in J])
    out["cos_resp"] = pair_cos([r["positions_text"] + " " + r["exp_skills_text"] for r in R],
                               [j["resp_text"] for j in J])
    out["cos_title"] = pair_cos([r["positions_text"] for r in R], [j["title_text"] for j in J])
    out["cos_objective"] = pair_cos([r["objective_text"] for r in R], [j["title_text"] + " " + j["resp_text"] for j in J])
    out["cos_edu"] = pair_cos([r["edu_text"] for r in R], [j["edu_req_text"] for j in J])

    ov = [skill_overlap(r["skills_list"], j["req_skills_list"]) for r, j in zip(R, J)]
    out["skill_coverage"] = [o[0] for o in ov]
    out["skill_precision"] = [o[1] for o in ov]
    out["skill_jaccard"] = [o[2] for o in ov]
    out["n_matched_skill_tokens"] = [len(_tok_set(r["skills_list"]) & _tok_set(j["req_skills_list"])) for r, j in zip(R, J)]

    out["n_skills"] = [len(r["skills_list"]) for r in R]
    out["resume_words"] = [len(t.split()) for t in rf]
    out["n_positions"] = [r["n_positions"] for r in R]
    out["n_certs"] = [r["n_certs"] for r in R]
    out["has_objective"] = [int(bool(r["objective_text"])) for r in R]
    out["exp_years"] = [r["exp_years"] for r in R]
    out["req_exp_min"] = [j["req_exp_min"] for j in J]
    out["req_exp_max"] = [j["req_exp_max"] for j in J]
    out["exp_gap"] = out["exp_years"] - out["req_exp_min"]
    out["n_req_skills"] = [len(j["req_skills_list"]) for j in J]
    out["job_words"] = [len(t.split()) for t in jf]
    return out[FEATURE_NAMES]


# ------------------------------------------------ full feature matrix (+LSA)
def build_feature_matrix(vec, svd, R, J) -> pd.DataFrame:
    """engineered pair features  +  LSA(resume)  +  LSA(job)  +  LSA(resume)*LSA(job)"""
    eng = pair_features_batch(vec, R, J)
    Rs = svd.transform(vec.transform([resume_full_text(r) for r in R]))
    Js = svd.transform(vec.transform([job_full_text(j) for j in J]))
    k = Rs.shape[1]
    parts = [
        eng.reset_index(drop=True),
        pd.DataFrame(Rs, columns=[f"res_lsa_{i}" for i in range(k)]),
        pd.DataFrame(Js, columns=[f"job_lsa_{i}" for i in range(k)]),
        pd.DataFrame(Rs * Js, columns=[f"prod_lsa_{i}" for i in range(k)]),
    ]
    return pd.concat(parts, axis=1)


def match_required_skills(resume_fields: dict, job_fields: dict, threshold: float = 0.5):
    """Which required-skill lines does the resume cover? (token overlap >= threshold)"""
    toks = set(resume_full_text(resume_fields).split())
    matched, missing = [], []
    for line in job_fields["req_skills_raw"]:
        lt = set(clean_text(line).split())
        if not lt:
            continue
        (matched if len(lt & toks) / len(lt) >= threshold else missing).append(line)
    return matched, missing


class ResumeJobMatcher:
    """Everything needed at inference time: TF-IDF -> LSA -> features -> regressor."""

    def __init__(self, vec, svd, model, feature_names):
        self.vec, self.svd, self.model, self.feature_names = vec, svd, model, list(feature_names)

    def features(self, R, J) -> pd.DataFrame:
        return build_feature_matrix(self.vec, self.svd, R, J)[self.feature_names]

    def predict(self, R, J) -> np.ndarray:
        return np.clip(self.model.predict(self.features(R, J)), 0.0, 1.0)
