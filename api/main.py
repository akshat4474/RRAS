from fastapi import FastAPI
from .routers import allocate, route, plan

app = FastAPI()

@app.get("/health")
def health(): return {"ok": True}

app.include_router(allocate.router)
app.include_router(route.router)
app.include_router(plan.router)
