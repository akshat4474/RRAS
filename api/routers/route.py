from fastapi import APIRouter, UploadFile, File, Form
from ..util import file_to_df
from core.routing import build_graph, shortest_path

router = APIRouter(prefix="/route", tags=["route"])

@router.post("/run")
async def run_route(
    roads: UploadFile = File(...),
    src: str = Form(...),
    dst: str = Form(...)
):
    roads_df = file_to_df(roads)
    G = build_graph(roads_df, alpha=1.0)
    route = shortest_path(G, src, dst)
    return {"route": route}
