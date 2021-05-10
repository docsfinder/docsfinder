from os.path import join

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from typer import Typer, echo

from .api.main import api

app = FastAPI(docs_url=None, redoc_url=None)

app.mount("/api", api)


@app.get("/", include_in_schema=False)
def index(request: Request):
    return RedirectResponse(join(request.url.path, "api", ""))


typer_app = Typer()


@typer_app.command()
def hello():
    echo("Hello World!")
