from os.path import join
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from ...core.full_document import FullDocument
from ...dependencies import dependencies

api = FastAPI(title="Docs Finder")


@api.get("/", include_in_schema=False, tags=["General"])
def index(request: Request):
    return RedirectResponse(join(request.url.path, "docs"))


@api.get("/query", tags=["General"], response_model=List[FullDocument])
def run(query: str):
    return dependencies.engine.find(query)
