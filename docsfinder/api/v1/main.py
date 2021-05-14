from os.path import join
from typing import List

from fastapi import FastAPI, Query, Request
from fastapi.responses import RedirectResponse

from ...core.models import FullDocument
from ...dependencies import dependencies

api = FastAPI(title="Docs Finder")


@api.get("/", include_in_schema=False, tags=["General"])
def index(request: Request):
    return RedirectResponse(join(request.url.path, "docs"))


@api.get("/query", tags=["General"], response_model=List[FullDocument])
def query(query: str):
    return dependencies.engine.find(query)


@api.get("/query-with-feedback", tags=["General"], response_model=List[FullDocument])
def query_with_feedback(
    query: str = Query(...),
    good_feedback: List[int] = Query(...),
    bad_feedback: List[int] = Query(...),
):
    return dependencies.engine.find_with_feedback(query, good_feedback, bad_feedback)
