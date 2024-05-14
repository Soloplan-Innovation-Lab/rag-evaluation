from uvicorn import Config, Server
from evaluation import evaluate
from fastapi import FastAPI
import nest_asyncio
from models import EvaluationPayload

# Apply nest_asyncio
nest_asyncio.apply()

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/evaluate")
async def read_item(item: EvaluationPayload):
    return evaluate(item)


def main():
    # ref: https://github.com/tiangolo/fastapi/issues/825#issuecomment-569826743
    config = Config(app=app, loop="asyncio")
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    main()
