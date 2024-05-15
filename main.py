from typing import List
from uvicorn import Config, Server
from evaluation import batch_evaluate
from fastapi import FastAPI
from models import EvaluationPayload

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/evaluate")
async def read_item(items: List[EvaluationPayload]):
    return batch_evaluate(items)


def main():
    # ref: https://github.com/tiangolo/fastapi/issues/825#issuecomment-569826743
    config = Config(app=app)
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    main()
