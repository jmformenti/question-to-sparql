import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class SparqlQueryGenerator():
  def __init__(self):
    prompt = PromptTemplate.from_template("""
    Given the next question and the info between brackets, write the SPARQL query for a wikibase:
    {question}
    Return only the SPARQL query enclosed by ```ruby without any explanation.
    """);

    if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] == '':
      del os.environ['OPENAI_API_BASE']

    if 'LLM_MODEL' in os.environ:
      llm = ChatOpenAI(model=os.environ['LLM_MODEL'], temperature=0.1)
    else:
      llm = ChatOpenAI(temperature=0.1)

    output_parser = StrOutputParser()

    self.chain = prompt | llm | output_parser

  def generate(self, text):
    return self.chain.invoke({"question": text})
