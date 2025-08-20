from fastapi import APIRouter, UploadFile, File
import pandas as pd
from ..util import file_to_df
from core.allocation import allocate_basic

router = APIRouter(prefix="/allocate", tags=["allocate"])

@router.post("/run")
async def run_allocate(areas: UploadFile = File(...)):
    areas_df = file_to_df(areas)
    # depots optional for global capacity; if not provided, we’ll scale to raw demand
    alloc = allocate_basic(areas_df, pd.DataFrame(columns=["capacity_food","capacity_water","capacity_meds"]))
    return {"allocation_table": alloc.to_dict(orient="records")}
