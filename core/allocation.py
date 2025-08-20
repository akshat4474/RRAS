import pandas as pd

def allocate_basic(areas_df: pd.DataFrame, depots_df: pd.DataFrame):
    """
    Heuristic allocator:
    - Base demand ∝ population * (0.5 + 0.5*severity/5)
    - Commodity-specific multipliers
    - Scales down to fit total depot capacities (global)
    Returns a DataFrame: area_id, food_units, water_units, med_kits, priority_score
    """
    df = areas_df.copy()
    if not {"area_id","population","severity"}.issubset(df.columns):
        raise ValueError("areas.csv missing required columns: area_id, population, severity")

    # Demand scores (0.5..1.0 multiplier for severity 0..5)
    sev_factor = 0.5 + 0.5 * (df["severity"].astype(float)/5.0)
    base = df["population"].astype(float) * sev_factor

    # Raw demands (tweak multipliers to taste)
    df["food_units_raw"]  = (base * 0.015).round().astype(int)
    df["water_units_raw"] = (base * 0.02 ).round().astype(int)
    df["med_kits_raw"]    = (df["severity"].astype(float) * 12).round().astype(int)

    # Capacities
    tot_food  = depots_df["capacity_food"].sum()   if "capacity_food"  in depots_df.columns else df["food_units_raw"].sum()
    tot_water = depots_df["capacity_water"].sum()  if "capacity_water" in depots_df.columns else df["water_units_raw"].sum()
    tot_meds  = depots_df["capacity_meds"].sum()   if "capacity_meds"  in depots_df.columns else df["med_kits_raw"].sum()

    # Scale to capacities (global proportional downscale if needed)
    def scale(col, cap):
        s = df[col].sum()
        if s <= 0 or cap <= 0: return 1.0
        return min(1.0, cap / s)

    sf = scale("food_units_raw",  tot_food)
    sw = scale("water_units_raw", tot_water)
    sm = scale("med_kits_raw",    tot_meds)

    df["food_units"]  = (df["food_units_raw"]  * sf).round().astype(int)
    df["water_units"] = (df["water_units_raw"] * sw).round().astype(int)
    df["med_kits"]    = (df["med_kits_raw"]    * sm).round().astype(int)

    # Priority score for scheduling
    # higher severity, higher population, (optionally) lower accessibility → higher priority
    acc = df["accessibility"] if "accessibility" in df.columns else 1.0
    df["priority_score"] = (df["severity"]*2 + (df["population"]/100000) - 0.5*acc).round(3)

    return df[["area_id","food_units","water_units","med_kits","priority_score"]]
