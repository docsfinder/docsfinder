from os.path import join

import typer
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from .api.main import api
from .core.engine import Engine

app = FastAPI(docs_url=None, redoc_url=None)

app.mount("/api", api)


@app.get("/", include_in_schema=False)
def index(request: Request):
    return RedirectResponse(join(request.url.path, "api", ""))


typer_app = typer.Typer()


@typer_app.command()
def find():
    typer.echo("Loading ...")
    engine = Engine()
    typer.echo("Loaded")
    while True:
        query = input("Enter a query: ")
        documents = engine.find(query)
        for (document, relevancy) in documents:
            typer.echo(
                f"Id: {document.id}, "
                + f"Relevancy: {relevancy}, "
                + f"Title: {document.title}",
            )
