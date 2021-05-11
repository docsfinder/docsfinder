import logging
from os.path import join

import typer
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .api.main import api
from .core.engine import Engine
from .dependencies import dependencies

app = FastAPI(docs_url=None, redoc_url=None)

app.mount("/api", api)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index(request: Request):
    return RedirectResponse(join(request.url.path, "api", ""))


@app.on_event("startup")
def startup_event():
    logging.info("Loading model ...")
    dependencies.engine = Engine(300)
    dependencies.engine.load()
    logging.info("Model loaded")


typer_app = typer.Typer()


@typer_app.command()
def find(data: str = "data/all_data.json"):
    typer.echo("Loading ...")
    engine = Engine(300)
    engine.load()
    typer.echo("Loaded")
    while True:
        query = input("Enter a query: ")
        documents = engine.find(query)
        for document in documents:
            typer.echo(
                f"Id: {document.id}, "
                + f"Relevancy: {document.relevancy}, "
                + f"Title: {document.title}",
            )


@typer_app.command()
def load():
    typer.echo("Loading ...")
    engine = Engine(300)
    engine.load()
    typer.echo("Loaded")


@typer_app.command()
def save(data: str = "data/all_data.json"):
    typer.echo("Loading ...")
    engine = Engine(300)
    engine.train(data)
    typer.echo("Loaded")
    engine.save()


@typer_app.command()
def test(
    data: str = "data/all_data.json",
    query: str = "data/all_query.json",
    top: int = 10,
):
    typer.echo("Loading ...")
    engine = Engine(300)
    engine.train(data)
    typer.echo("Loaded")
    typer.echo("Running precision test ...")
    precision = engine.test_precision(query, top)
    typer.echo(f"Precision: {precision}")
    typer.echo("Running recall test ...")
    recall = engine.test_recall(query, top)
    typer.echo(f"Recall: {recall}")
