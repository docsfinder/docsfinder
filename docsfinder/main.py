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
    dependencies.engine = Engine()
    dependencies.engine.load()
    logging.info("Model loaded")


typer_app = typer.Typer()


@typer_app.command()
def find():
    typer.echo("Loading ...")
    engine = Engine()
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
    engine = Engine()
    engine.load()
    typer.echo("Loaded")


@typer_app.command()
def save(data: str):
    typer.echo("Loading ...")
    engine = Engine()
    engine.train(data)
    typer.echo("Loaded")
    engine.save()


@typer_app.command()
def save_all():
    save("data/all_data.json")


@typer_app.command()
def save_cisi():
    save("data/cisi_data.json")


@typer_app.command()
def save_cran():
    save("data/cran_data.json")


@typer_app.command()
def test(data: str, query: str, top: int = 10):
    typer.echo("Loading ...")
    engine = Engine()
    engine.train(data)
    typer.echo("Loaded")
    typer.echo("Running Precision test ...")
    precision = engine.test_precision(query, top)
    typer.echo(f"Precision: {precision}")
    typer.echo("Running Recall test ...")
    recall = engine.test_recall(query, top)
    typer.echo(f"Recall: {recall}")
    typer.echo("Running F test ...")
    f = engine.test_f(precision=precision, recall=recall)
    typer.echo(f"F: {f}")
    typer.echo("Running Fallout test ...")
    fallout = engine.test_recall(query, top)
    typer.echo(f"Fallout: {fallout}")


@typer_app.command()
def test_all(top: int = 10):
    test("data/all_data.json", "data/all_query.json", top)


@typer_app.command()
def test_cisi(top: int = 10):
    test("data/cisi_data.json", "data/cisi_query.json", top)


@typer_app.command()
def test_cran(top: int = 10):
    test("data/cran_data.json", "data/cran_query.json", top)
