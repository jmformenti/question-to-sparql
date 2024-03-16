import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class SparqlQueryGenerator():
  def __init__(self):
    self.prompt = PromptTemplate.from_template("""
    Given the next question and the info between brackets, write the SPARQL query for a wikibase:
    {question}
    Return only the SPARQL query enclosed by ```ruby without any explanation.
    """);

    if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] == '':
      del os.environ['OPENAI_API_BASE']

    if 'LLM_MODEL' in os.environ:
      self.llm = ChatOpenAI(model=os.environ['LLM_MODEL'], temperature=0.1)
    else:
      self.llm = ChatOpenAI(temperature=0.1)

  def generate(self, text):
    return self.llm.invoke(self.prompt.format(question=text))
