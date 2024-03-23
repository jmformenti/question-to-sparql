import logging
import re

from urllib.parse import quote

from api.model import ChatCompletionRequest, ChatCompletionStreamResponse, ChatCompletionResponseStreamChoice, DeltaMessage, ChatMessage
from service.enrich_question import QuestionEnricher
from service.generate_sparql import SparqlQueryGenerator


logger = logging.getLogger(f'app.{__name__}')

class QuestionToSparql:
  EXTRACT_QUERY_REGEX = r"```.*?```"

  def __init__(self):
    self.question_enricher = QuestionEnricher()
    self.sparql_query_generator = SparqlQueryGenerator()

  def _extractQuery(self, text):
    matches = re.findall(self.EXTRACT_QUERY_REGEX, text, re.DOTALL)
    return matches[0].strip("```").strip('ruby')

  def _create_stream_response(self, text, request):
    return ChatCompletionStreamResponse(
      choices=[
        ChatCompletionResponseStreamChoice(
          index=0,
          delta=DeltaMessage(content=text, role='assistant')
        )
      ],
      model=request.model
    )

  def generate_answer(self, request: ChatCompletionRequest, debug=False):
    if request.stream:
      question = request.messages[-1].content
      logger.info(f'# Original question\n{question}')
      
      enriched_text, enriched_text_ui = self.question_enricher.enrich(question, debug)
      logger.info(f'\n# Enriched question\n{enriched_text}\n{enriched_text_ui}')
      yield f'data: {self._create_stream_response(f'<b>Enriched text</b>: {enriched_text_ui}\n\n', request).model_dump_json()}\n\n'

      sparql_query = self.sparql_query_generator.generate(enriched_text).strip()
      logger.info(f'\n# SPARQL query\n{sparql_query}')
      yield f'data: {self._create_stream_response(f'<b>SPARQL query</b>:\nopen <a target="_blank" href="https://query.wikidata.org/#{quote(self._extractQuery(sparql_query))}">here</a>\n{sparql_query}', request).model_dump_json()}\n\n'

      yield 'data: [DONE]\n\n'

      return
    else:
      raise NotImplementedError()
