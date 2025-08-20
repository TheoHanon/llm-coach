import pandas as pd
import numpy as np
import json, pathlib, os
from typing import Dict, Any, List, Tuple


from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain.schema import Document

_INDEX_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "index_faiss"
_INDEX = None
_RETRIEVER = None

def _get_retriever(k : int = 4):
    global _INDEX, _RETRIEVER
    if _RETRIEVER is None:
        emb = MistralAIEmbeddings(model="mistral-embed")
        _INDEX = FAISS.load_local(str(_INDEX_DIR), emb, allow_dangerous_deserialization=True)
        _RETRIEVER = _INDEX.as_retriever(search_kwargs={"k": k, "fetch_k" : 20}, search_type = "mmr")
    return _RETRIEVER


def _retrieve(query : str, k : int = 4) -> Tuple[List[Document], List[Dict[str, Any]], str]:
    retriever = _get_retriever(k)
    docs: List[Document] = retriever.invoke(query)
    bib, ctx = [], []

    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title") or d.metadata.get("filename") or f"Source {i}"
        page  = d.metadata.get("page")
        bib.append({"id": i, "title": title, "page": page, "category": d.metadata.get("category")})
        ctx.append(f"[{i}] {d.page_content}")

    return docs, bib, "\n\n".join(ctx)


def _build_query(specs : Dict[str, str]) -> str:

    sport = specs.get("sport", "endurance sports")
    goal = specs.get("goal", "build_base")
    add_rem = specs.get("additional_remarks", "none")


    return (
        f"{sport} training best practices for goal being {goal}\n"
        f"Provide recent, evidence-based guidance.\n"
        f"Additional specification : {add_rem}\n"
    )

def _get_fitness_summary(snapshot: Dict[str, Any]) -> str:

    if not snapshot.get("result"):
        return json.dumps({"status": "No data found"})

    activity_list = snapshot["result"]["SnapshotFitnessDetails"]["payload"]["activityList"]
    if not activity_list:
        return json.dumps({"status": "No activities"})

    df = pd.DataFrame(activity_list)

    
    want_cols = ["startTimeLocal","activityType","distance","duration","intensityFactor",
                 "trainingStressScore","activityTrainingLoad"]
    df = df[[c for c in want_cols if c in df.columns]].copy()
    df["activityType"] = df["activityType"].apply(lambda x : x["typeKey"])

    
    if "startTimeLocal" in df.columns:
        df["start_dt"] = pd.to_datetime(df["startTimeLocal"], errors="coerce")
    else:
        df["start_dt"] = pd.NaT

    # numerics
    for c in ["distance","duration","intensityFactor","trainingStressScore","activityTrainingLoad"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

    # derived
    df["distance_km"] = (df["distance"]/1000.0) if "distance" in df else np.nan
    df["minutes"] = (df["duration"]/60.0) if "duration" in df else np.nan
    df["speed_kmh"] = (df["distance_km"]/(df["minutes"]/60.0))

    
    if df["start_dt"].notna().any():
        min_dt, max_dt = df["start_dt"].min(), df["start_dt"].max()
        time_gap_days = int(max(1, (max_dt - min_dt).days))  # ensure >=1 to avoid div by zero
    else:
        time_gap_days = None

    # totals
    totals = {
        "sessions": int(len(df)),
        "hours": round(float(np.nansum(df.get("minutes", 0)))/60.0, 2),
        "distance_km": round(float(np.nansum(df.get("distance_km", 0))), 1),
        "total_training_stress_score": round(float(np.nansum(df.get("trainingStressScore", 0))), 0) if "trainingStressScore" in df else None,
        "total_training_load": round(float(np.nansum(df.get("activityTrainingLoad", 0))), 0) if "activityTrainingLoad" in df else None,
        "avg_intensity_factor": round(float(np.nanmean(df["intensityFactor"])), 3) if "intensityFactor" in df and df["intensityFactor"].notna().any() else None,
        "time_gap_days": time_gap_days,
    }


    if time_gap_days:
        weeks = time_gap_days / 7.0
        per_week = {
            "sessions_per_week": round(totals["sessions"] / weeks, 1),
            "hours_per_week": round(totals["hours"] / weeks, 1),
            "distance_km_per_week": round(totals["distance_km"] / weeks, 1),
            "training_stress_score_per_week": round(totals["total_training_stress_score"] / weeks, 1) if totals.get("total_training_stress_score") else None,
            "trainingLoad_per_week": round(totals["total_trainingLoad"] / weeks, 1) if totals.get("total_trainingLoad") else None,
        }
    else:
        per_week = {}

    # per-sport (session averages)
    per_sport = {}
    if "activityType" in df:
        for sport, g in df.groupby("activityType"):
            per_sport[sport] = {
                "avg_distance_km": round(float(np.nanmean(g["distance_km"])), 2) if g["distance_km"].notna().any() else None,
                "avg_session_min": round(float(np.nanmean(g["minutes"])), 1) if g["minutes"].notna().any() else None,
                "avg_speed_kmh": round(float(np.nanmean(g["speed_kmh"])), 2) if g["speed_kmh"].notna().any() else None,
                "avg_intensity_factor": round(float(np.nanmean(g["intensityFactor"])), 3) if "intensityFactor" in g and g["intensityFactor"].notna().any() else None,
                "avg_training_stress_score": round(float(np.nanmean(g["trainingStressScore"])), 1) if "trainingStressScore" in g and g["trainingStressScore"].notna().any() else None,
                "avg_training_load": round(float(np.nanmean(g["activityTrainingLoad"])), 1) if "activityTrainingLoad" in g and g["activityTrainingLoad"].notna().any() else None,
                "sessions": int(len(g)),
            }

    # primary sport & focus cue
    if "activityType" in df:
        dist_by = df.groupby("activityType")["distance_km"].sum(min_count=1).fillna(0.0)
        freq_by = df.groupby("activityType").size()
        primary_sport = (dist_by.idxmax() if dist_by.max() > 0 else freq_by.idxmax())
        avg_if = totals.get("avg_intensity_factor")
    else:
        primary_sport, avg_if = None, None


    summary = {
        "status": "ok",
        "totals": totals,
        "per_week": per_week,
        "per_sport": per_sport,
        "primary_sport": primary_sport,
    }

    return json.dumps(summary)