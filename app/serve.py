import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from api.model import ChatCompletionRequest
from service.question_to_sparql import QuestionToSparql


logger = logging.getLogger('app')
logger.setLevel(logging.WARN)
logger.addHandler(logging.StreamHandler())

app = FastAPI()
question_to_sparql = QuestionToSparql()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
  if request.stream:
    return StreamingResponse(content=question_to_sparql.generate_answer(request), media_type="text/event-stream")
  else:
    return JSONResponse(content=question_to_sparql.generate_answer(request))

if __name__ == "__main__":
  os.environ['OPENAI_API_BASE'] = os.environ['OPENAI_API_BASE'].replace('host.docker.internal', 'localhost')
  uvicorn.run("serve:app", host="0.0.0.0", port=9000, reload=True)