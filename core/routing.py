import math
import pandas as pd
import networkx as nx

INF = 10**9

def build_graph(roads_df: pd.DataFrame, alpha: float = 1.0):
    """
    Returns a DiGraph with edge attributes:
      dist (km), risk (int), weight (dist + alpha*risk) or INF if blocked
    """
    G = nx.DiGraph()
    req = {"from_node","to_node","distance_km","is_blocked","risk_level"}
    # roads.csv in your data uses these column names already
    if not req.issubset(roads_df.columns):
        # try alternative header set (edge_id, from_node, to_node, ...)
        req_alt = {"edge_id","from_node","to_node","distance_km","is_blocked","risk_level"}
        if not req_alt.issubset(roads_df.columns):
            raise ValueError("roads.csv missing required columns.")
    for _, r in roads_df.iterrows():
        u = str(r["from_node"]); v = str(r["to_node"])
        d = float(r["distance_km"])
        risk = int(r.get("risk_level", 0))
        blocked = bool(r.get("is_blocked", False))
        w = INF if blocked else d + alpha * risk
        G.add_edge(u, v, dist=d, risk=risk, blocked=blocked, weight=w)
        # assume bidirectional unless your data says otherwise
        G.add_edge(v, u, dist=d, risk=risk, blocked=blocked, weight=w)
    return G

def shortest_path(G, src: str, dst: str):
    try:
        path = nx.shortest_path(G, src, dst, weight="weight")
        edges = list(zip(path[:-1], path[1:]))
        dist = sum(G[u][v]["dist"] for u, v in edges)
        # naive ETA: 30 km/h
        eta_min = (dist / 30.0) * 60.0
        return {"path_nodes": path, "total_km": round(dist,2), "est_time_min": round(eta_min,1)}
    except nx.NetworkXNoPath:
        return {"path_nodes": [], "total_km": None, "est_time_min": None, "error": "No path"}
