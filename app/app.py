import os
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import folium
import streamlit.components.v1 as components

# --- OSRM helpers ---
import polyline
from functools import lru_cache

load_dotenv()
API = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60

st.set_page_config(page_title="RRAS", layout="wide")
st.title("RRAS Dashboard")

# ---------------------------
# Session state
# ---------------------------
if "route_result" not in st.session_state:
    st.session_state.route_result = None
if "plan_result" not in st.session_state:
    st.session_state.plan_result = None

# map html caches (Fix B)
if "route_map_key" not in st.session_state:
    st.session_state.route_map_key = None
if "route_map_html" not in st.session_state:
    st.session_state.route_map_html = None
if "plan_map_key" not in st.session_state:
    st.session_state.plan_map_key = None
if "plan_map_html" not in st.session_state:
    st.session_state.plan_map_html = None

# ---------------------------
# Helpers
# ---------------------------
def df_from_upload(f):
    if f is None:
        return None
    return pd.read_csv(f)

def make_node_coords(areas_df, depots_df):
    coords = {}
    if areas_df is not None and {"area_id", "lat", "lon"}.issubset(areas_df.columns):
        for _, r in areas_df.iterrows():
            coords[str(r["area_id"])] = (float(r["lat"]), float(r["lon"]))
    if depots_df is not None and {"depot_id", "lat", "lon"}.issubset(depots_df.columns):
        for _, r in depots_df.iterrows():
            coords[str(r["depot_id"])] = (float(r["lat"]), float(r["lon"]))
    return coords

def severity_color(sev: int):
    return {0: "green", 1: "green", 2: "lightblue", 3: "blue", 4: "orange", 5: "red"}.get(int(sev), "blue")

def folium_base_map(node_coords, areas_df, depots_df):
    """Markers + legend only (no route lines)."""
    if node_coords:
        lats = [lt for lt, _ in node_coords.values()]
        lons = [ln for _, ln in node_coords.values()]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))
    else:
        center = (20.5937, 78.9629)  # fallback: India center

    m = folium.Map(location=center, zoom_start=10, control_scale=True)

    # --- depots ---
    if depots_df is not None and not depots_df.empty:
        for _, r in depots_df.iterrows():
            did = str(r["depot_id"])
            lat, lon = node_coords.get(did, (None, None))
            if lat is None:
                continue
            popup_html = f"""
            <b>Depot {did}</b><br>
            Food: {r.get('capacity_food', 'N/A')}<br>
            Water: {r.get('capacity_water', 'N/A')}<br>
            Medkits: {r.get('capacity_meds', r.get('capacity_med', 'N/A'))}
            """
            folium.Marker(
                [lat, lon],
                tooltip=f"Depot {did}",
                popup=popup_html,
                icon=folium.Icon(color="green", icon="truck", prefix="fa"),
            ).add_to(m)

    # --- areas ---
    if areas_df is not None and not areas_df.empty:
        for _, r in areas_df.iterrows():
            aid = str(r["area_id"])
            lat, lon = node_coords.get(aid, (None, None))
            if lat is None:
                continue
            sev = int(r.get("severity", 0))
            popup_html = f"""
            <b>Area {aid}</b><br>
            Severity: {sev}<br>
            Population: {r.get('population', 'N/A')}
            """
            folium.CircleMarker(
                [lat, lon],
                radius=6,
                tooltip=f"Area {aid} | Sev {sev}",
                popup=popup_html,
                color=severity_color(sev),
                fill=True,
                fill_opacity=0.8,
            ).add_to(m)

    # --- legend ---
    legend_html = """
     <div style="position: fixed; 
                 bottom: 30px; left: 30px; width: 180px; height: 150px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white; padding: 10px;">
     <b>Severity Legend</b><br>
     <i style="background:green; width:10px; height:10px; float:left; margin-right:5px"></i> Low (0-1)<br>
     <i style="background:lightblue; width:10px; height:10px; float:left; margin-right:5px"></i> Moderate (2)<br>
     <i style="background:blue; width:10px; height:10px; float:left; margin-right:5px"></i> Serious (3)<br>
     <i style="background:orange; width:10px; height:10px; float:left; margin-right:5px"></i> Severe (4)<br>
     <i style="background:red; width:10px; height:10px; float:left; margin-right:5px"></i> Critical (5)<br>
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def map_html(m):
    """Return standalone HTML for a Folium map (Fix B)."""
    return m._repr_html_()

def download_btn(df: pd.DataFrame, label: str, filename: str, key: str):
    if df is None or df.empty:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
        type="secondary",
    )

def files_bytes(**named_files):
    """Build multipart files payload from Streamlit uploads (raw bytes)."""
    out = {}
    for field, f in named_files.items():
        if f is None:
            continue
        out[field] = (f.name or f"{field}.csv", f.getvalue(), "text/csv")
    return out

# ---------------------------
# OSRM road-following helpers
# ---------------------------
@lru_cache(maxsize=1024)
def osrm_seg(lat1, lon1, lat2, lon2):
    """Get road-following segment coordinates from OSRM (cached)."""
    url = (
        "http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=polyline"
    )
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "routes" in data and data["routes"]:
            return tuple(polyline.decode(data["routes"][0]["geometry"]))  # tuple for cacheability
    except Exception as e:
        st.warning(f"OSRM segment failed (fallback to straight line): {e}")
    return ((lat1, lon1), (lat2, lon2))

def osrm_path_for_nodes(nodes, node_coords):
    coords_seq = []
    for i in range(len(nodes) - 1):
        n1, n2 = nodes[i], nodes[i + 1]
        if n1 in node_coords and n2 in node_coords:
            lat1, lon1 = node_coords[n1]
            lat2, lon2 = node_coords[n2]
            seg = osrm_seg(lat1, lon1, lat2, lon2)
            coords_seq.extend(seg if i == 0 else seg[1:])
    return coords_seq

def folium_route_map_osrm(node_coords, areas_df, depots_df, path_nodes):
    """Single-route map with OSRM-snapped polyline."""
    m = folium_base_map(node_coords, areas_df, depots_df)
    if isinstance(path_nodes, list) and len(path_nodes) >= 2:
        coords_seq = osrm_path_for_nodes(path_nodes, node_coords)
        if len(coords_seq) >= 2:
            folium.PolyLine(coords_seq, weight=5, opacity=0.9, color="blue").add_to(m)
    return m

def folium_plan_map_osrm(node_coords, areas_df, depots_df, trips):
    """Plan map with OSRM-snapped polylines; each trip colored."""
    m = folium_base_map(node_coords, areas_df, depots_df)
    colors = ["blue", "red", "green", "purple", "orange", "black", "cadetblue", "darkred"]
    for idx, trip in enumerate(trips):
        nodes = trip.get("path_nodes") or []
        if len(nodes) < 2:
            continue
        coords_seq = osrm_path_for_nodes(nodes, node_coords)
        if len(coords_seq) >= 2:
            color = colors[idx % len(colors)]
            tooltip = f"{trip.get('trip_id','Trip')} | {trip.get('total_km','?')} km, {trip.get('est_time_min','?')} min"
            folium.PolyLine(coords_seq, weight=5, opacity=0.9, color=color, tooltip=tooltip).add_to(m)
    return m

# ---------------------------
# Health check
# ---------------------------
st.caption("Health check")
try:
    st.json(requests.get(f"{API}/health", timeout=5).json())
except Exception as e:
    st.error(f"API not reachable at {API}: {e}")

st.divider()

# ---------------------------
# Uploads
# ---------------------------
st.subheader("Upload Data")
c1, c2, c3 = st.columns(3)
with c1:
    areas_file = st.file_uploader("areas.csv", type="csv")
with c2:
    depots_file = st.file_uploader("depots.csv", type="csv")
with c3:
    roads_file = st.file_uploader("roads.csv", type="csv")

areas_df = df_from_upload(areas_file)
depots_df = df_from_upload(depots_file)
roads_df = df_from_upload(roads_file)
node_coords = make_node_coords(areas_df, depots_df)

# ---------------------------
# Allocation
# ---------------------------
st.subheader("Predict Allocation")
if st.button("Run Allocation") and areas_file:
    try:
        r = requests.post(f"{API}/allocate/run", files=files_bytes(areas=areas_file), timeout=TIMEOUT)
        payload = r.json()
        alloc_df = pd.DataFrame(payload.get("allocation_table", []))
        if alloc_df.empty:
            st.warning("Allocation returned empty table.")
        st.dataframe(alloc_df, use_container_width=True)
        download_btn(alloc_df, "Download allocation_table.csv", "allocation_table.csv", "dl_alloc")
    except ValueError:
        st.error(f"Non-JSON response. Status={r.status_code}. Body={r.text[:800]}")
    except Exception as e:
        st.error(f"Allocation failed: {e}")

st.divider()

# ---------------------------
# Single Route
# ---------------------------
st.subheader("Compute Single Route")

colA, colB = st.columns(2)
with colA:
    depot_node = st.selectbox(
        "Depot ID",
        depots_df["depot_id"].astype(str).tolist()
        if depots_df is not None and "depot_id" in depots_df.columns
        else [],
        index=0 if depots_df is not None and not depots_df.empty else None,
        placeholder="Select a depot",
    )
with colB:
    area_node = st.selectbox(
        "Area ID",
        areas_df["area_id"].astype(str).tolist()
        if areas_df is not None and "area_id" in areas_df.columns
        else [],
        index=0 if areas_df is not None and not areas_df.empty else None,
        placeholder="Select an area",
    )

if st.button("Run Route"):
    if not all([areas_file, depots_file, roads_file]):
        st.warning("Please upload all three CSVs.")
    elif not depot_node or not area_node:
        st.warning("Select both Depot and Area.")
    else:
        try:
            r = requests.post(
                f"{API}/route/run",
                files=files_bytes(areas=areas_file, depots=depots_file, roads=roads_file),
                data={"src": depot_node, "dst": area_node},
                timeout=TIMEOUT,
            )
            resp = r.json()
            route_json = resp.get("route", resp)

            # Store result in session state
            st.session_state.route_result = {
                "route_json": route_json,
                "depot_node": depot_node,
                "area_node": area_node,
            }

            st.json(route_json)

        except ValueError:
            st.error(f"Non-JSON response. Status={r.status_code}. Body={r.text[:800]}")
        except Exception as e:
            st.error(f"Route failed: {e}")

# Route map (Fix B + OSRM)
if st.session_state.route_result and node_coords:
    route_json = st.session_state.route_result["route_json"]
    depot_node = st.session_state.route_result["depot_node"]
    area_node = st.session_state.route_result["area_node"]

    if isinstance(route_json.get("path_nodes"), list):
        path = route_json["path_nodes"]

        # Cache key to rebuild only when inputs change
        route_key = ("route_osrm", tuple(path), tuple(sorted(node_coords.items())))
        if st.session_state.route_map_key != route_key:
            st.session_state.route_map_key = route_key
            m = folium_route_map_osrm(node_coords, areas_df, depots_df, path)
            st.session_state.route_map_html = map_html(m)

        st.caption("Route map (OSRM road-following)")
        components.html(st.session_state.route_map_html, height=560, scrolling=False)

        # Download row
        row = pd.DataFrame(
            [
                {
                    "depot_id": depot_node,
                    "area_id": area_node,
                    "path": " -> ".join(path),
                    "total_km": route_json.get("total_km"),
                    "est_time_min": route_json.get("est_time_min"),
                }
            ]
        )
        download_btn(row, "Download route_table.csv", "route_table.csv", "dl_route_single")

st.divider()

# ---------------------------
# Full Plan
# ---------------------------
st.subheader("Generate Full Plan")
if st.button("Run Full Plan"):
    if not all([areas_file, depots_file, roads_file]):
        st.warning("Please upload all three CSVs.")
    else:
        try:
            r = requests.post(
                f"{API}/plan/run",
                files=files_bytes(areas=areas_file, depots=depots_file, roads=roads_file),
                timeout=TIMEOUT,
            )
            payload = r.json()
            trips = payload.get("trips", [])
            trips_df = pd.DataFrame(trips)

            # Store result
            st.session_state.plan_result = {"trips_df": trips_df, "payload": payload}

            if trips_df.empty:
                st.warning("Plan returned no trips.")
            else:
                st.dataframe(trips_df, use_container_width=True)

                # quick impact metrics
                total_km = trips_df["total_km"].fillna(0).sum()
                total_trips = len(trips_df)
                eta_sum = trips_df["est_time_min"].fillna(0).sum()
                st.info(f"🚚 {total_trips} trips | 🛣️ {total_km:.1f} km | ⏱️ {eta_sum:.0f} min est.")

        except ValueError:
            st.error(f"Non-JSON response. Status={r.status_code}. Body={r.text[:800]}")
        except Exception as e:
            st.error(f"Plan failed: {e}")

# Plan map (Fix B + OSRM + colored trips)
if st.session_state.plan_result and node_coords:
    trips_df = st.session_state.plan_result["trips_df"]

    if trips_df is not None and not trips_df.empty and "path_nodes" in trips_df.columns:
        trips = trips_df.to_dict(orient="records")

        # Cache key from all paths + node coords
        plan_key = (
            "plan_osrm",
            tuple(tuple(p) for p in trips_df["path_nodes"] if isinstance(p, list)),
            tuple(sorted(node_coords.items())),
        )
        if st.session_state.plan_map_key != plan_key:
            st.session_state.plan_map_key = plan_key
            m = folium_plan_map_osrm(node_coords, areas_df, depots_df, trips)
            st.session_state.plan_map_html = map_html(m)

        st.caption("Plan map (OSRM road-following, colored by trip)")
        components.html(st.session_state.plan_map_html, height=560, scrolling=False)

        # Flatten & download
        trips_flat = trips_df.copy()
        trips_flat["path"] = trips_flat["path_nodes"].apply(
            lambda xs: " -> ".join(xs) if isinstance(xs, list) else ""
        )
        trips_flat = trips_flat.drop(columns=["path_nodes"])
        download_btn(trips_flat, "Download route_table.csv", "route_table.csv", "dl_route_plan")
