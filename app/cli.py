import logging
import argparse
from api.model import ChatCompletionRequest, ChatMessage
from service.question_to_sparql import QuestionToSparql

logger = logging.getLogger('app')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser(description='SPARQL query generator.')
parser.add_argument('--question', type=str, required=True, help='Your question.')
parser.add_argument('--debug', action='store_true', help='Show debug info.')

args = parser.parse_args()

question_to_sparql = QuestionToSparql()
request = ChatCompletionRequest(stream=True, model='ignored', messages=[ChatMessage(role='ignored', content=args.question)])
for _ in question_to_sparql.generate_answer(request, args.debug):
  pass

logger.info('\ndone.\n')