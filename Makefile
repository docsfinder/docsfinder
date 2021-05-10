HOST = localhost
PORT = 8000

install:
	poetry install

tests: install
	poetry run flake8 . --count --show-source --statistics --max-line-length=88 --extend-ignore=E203
	poetry run black . --check
	poetry run isort . --profile=black
	poetry run pytest --cov=./ --cov-report=xml

export:
	poetry export -f requirements.txt -o requirements.txt --without-hashes

export_and_commit: export
	git config user.name 'docsfinder'
	git config user.email 'codestrangeofficial@gmail.com'
	git add requirements.txt
	git commit --allow-empty -m "Update requirements.txt"
	git push

update_index:
	cp README.md docs/index.md

update_index_and_commit: update_index
	git config user.name 'docsfinder'
	git config user.email 'codestrangeofficial@gmail.com'
	git add docs/index.md
	git commit --allow-empty -m "Update docs/index.md"
	git push

run: install
	poetry run uvicorn docsfinder.main:app --reload --host ${HOST} --port ${PORT}

build:
	docker build -t docsfinder:latest .

deploy:
	docker run -d -p 8000:80 --name docsfinder-container --env-file .env docsfinder:latest

rmcontainer:
	docker container rm docsfinder-container --force

rmimage:
	docker image rm docsfinder:latest

build_deploy: build deploy

rmall: rmcontainer rmimage

redeploy: rmall build_deploy

logs:
	docker logs docsfinder-container
