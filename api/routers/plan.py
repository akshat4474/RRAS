# api/routers/plan.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import traceback
from ..util import file_to_df
from core.allocation import allocate_basic
from core.routing import build_graph, shortest_path

router = APIRouter(prefix="/plan", tags=["plan"])

REQ_AREAS = {"area_id", "lat", "lon", "population", "severity"}
REQ_DEPOTS = {"depot_id", "lat", "lon"}
REQ_ROADS_MAIN = {"from_node", "to_node", "distance_km", "is_blocked", "risk_level"}
REQ_ROADS_ALT  = {"edge_id", "from_node", "to_node", "distance_km", "is_blocked", "risk_level"}

def ensure_columns(df: pd.DataFrame, needed: set, name: str):
    if not needed.issubset(df.columns):
        missing = list(needed - set(df.columns))
        raise HTTPException(status_code=400, detail=f"{name} missing columns: {missing}")

@router.post("/run")
async def run_plan(
    areas: UploadFile = File(...),
    depots: UploadFile = File(...),
    roads: UploadFile = File(...),
):
    try:
        areas_df  = file_to_df(areas)
        depots_df = file_to_df(depots)
        roads_df  = file_to_df(roads)

        # --- schema checks ---
        ensure_columns(areas_df, REQ_AREAS, "areas.csv")
        ensure_columns(depots_df, REQ_DEPOTS, "depots.csv")
        if not (REQ_ROADS_MAIN.issubset(roads_df.columns) or REQ_ROADS_ALT.issubset(roads_df.columns)):
            raise HTTPException(status_code=400, detail="roads.csv missing columns; need either "
                                f"{sorted(list(REQ_ROADS_MAIN))} or {sorted(list(REQ_ROADS_ALT))}")

        # --- allocation ---
        alloc_df = allocate_basic(areas_df, depots_df)

        # --- prioritization ---
        merged = areas_df.merge(alloc_df, on="area_id", how="left")
        merged = merged.sort_values("priority_score", ascending=False)

        # --- choose nearest depot (beeline) ---
        def nearest_depot(row):
            best, bestd = None, float("inf")
            alat, alon = float(row["lat"]), float(row["lon"])
            for _, d in depots_df.iterrows():
                dlat, dlon = float(d["lat"]), float(d["lon"])
                dd = (dlat - alat) ** 2 + (dlon - alon) ** 2
                if dd < bestd:
                    bestd, best = dd, str(d["depot_id"])
            return best

        merged["depot_id"] = merged.apply(nearest_depot, axis=1)

        # --- build graph (supports alt header set) ---
        if REQ_ROADS_ALT.issubset(roads_df.columns):
            roads_use = roads_df.rename(columns={"edge_id":"edge_id",
                                                 "from_node":"from_node",
                                                 "to_node":"to_node",
                                                 "distance_km":"distance_km",
                                                 "is_blocked":"is_blocked",
                                                 "risk_level":"risk_level"})
        else:
            roads_use = roads_df

        G = build_graph(roads_use, alpha=1.0)

        # --- route each prioritized area ---
        trips = []
        for _, r in merged.iterrows():
            depot = str(r["depot_id"]); area = str(r["area_id"])
            route = shortest_path(G, depot, area)
            trips.append({
                "trip_id": f"{depot}->{area}",
                "depot_id": depot,
                "area_id": area,
                "path_nodes": route.get("path_nodes"),
                "total_km": route.get("total_km"),
                "est_time_min": route.get("est_time_min"),
                "food_units": int(r.get("food_units", 0)) if pd.notna(r.get("food_units", 0)) else 0,
                "water_units": int(r.get("water_units", 0)) if pd.notna(r.get("water_units", 0)) else 0,
                "med_kits": int(r.get("med_kits", 0)) if pd.notna(r.get("med_kits", 0)) else 0,
                "priority_score": float(r.get("priority_score", 0.0)) if pd.notna(r.get("priority_score", 0.0)) else 0.0,
            })

        return {"trips": trips, "allocation_table": alloc_df.to_dict(orient="records")}

    except HTTPException:
        raise
    except Exception as e:
        # Log server-side and return helpful message
        print("PLAN ERROR:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
