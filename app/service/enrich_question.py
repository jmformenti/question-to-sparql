import os
import json
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from service.wikibase_connector import WikibaseConnector
from string import Template


logger = logging.getLogger(f'app.{__name__}')

class GraphState(TypedDict):
  """
  Represents the state of our graph.

  Attributes:
      keys: A dictionary where each key is a string.
  """

  keys: Dict[str, any]

class DetectedItems(BaseModel):
  P: str = Field(description="List of substrings for P items")
  Q: str = Field(description="List of substrings for P items")

class QuestionEnricher:
  PROMPT_TASK = """
Identify in the input text the possible substrings for a P and Q items in a Wikibase.
Do not include different concepts for a single Q item.
"""
  PROMPT_VALID_EXAMPLES = """
Input:
How many children had J.S. Bach?
Output:
```
{{
"P": ["children"],
"Q": ["J.S. Bach"]
}}
```
"""
  PROMPT_WITH_NO_ERROR = """
{task}
{format_instructions}
<< Examples >>
{valid_examples}
<< End Examples >>
Input:
{input}
"""
  PROMPT_WITH_ERROR = """
{task}
{format_instructions}
<< Examples >>
Valid examples:
{valid_examples}
Invalid examples:
{invalid_examples}
<< End Examples >>
Input:
{input}
"""
  MAX_ATTEMPTS=3
  ITEM_CONTEXT_TEMPLATE = Template('${text} (${number})')
  ITEM_CONTEXT_UI_TEMPLATE = Template('<a target="_blank" href="https://www.wikidata.org/wiki/${type}:${number}">${text}</a>')

  def __init__(self):
    self.wikibase = WikibaseConnector()
    self.parser = JsonOutputParser(pydantic_object=DetectedItems)
    self.llm = self._get_llm()
    self.workflow = self._get_workflow()
    self.workflow_config = {'recursion_limit': 50 }

  def _get_workflow(self):
    workflow = StateGraph(GraphState)

    workflow.add_node('detect_p_q_items', self._detect_p_q_items)

    workflow.set_entry_point('detect_p_q_items')
    workflow.add_conditional_edges(
      'detect_p_q_items',
      self._decide_valid_items,
      {
        'detect_p_q_items': 'detect_p_q_items',
        'end': END
      }
    )

    return workflow.compile()

  def _get_llm(self):
    if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] == '':
      del os.environ['OPENAI_API_BASE']

    if 'LLM_MODEL' in os.environ:
      return ChatOpenAI(model=os.environ['LLM_MODEL'], temperature=0.1)
    else:
      return ChatOpenAI(temperature=0.1)

  def _item_not_in_input(self, detected_items, input):
    for item in detected_items['P']:
      if item not in input:
        return item
    for item in detected_items['Q']:
      if item not in input:
        return item
    return None

  def _get_prompt(self, state: GraphState):
    state_dict = state['keys']

    prompt = None
    if 'error' in state_dict:
      prompt = PromptTemplate(template=self.PROMPT_WITH_ERROR,
        input_variables=["input"],
        partial_variables={
          'task': self.PROMPT_TASK,
          'format_instructions': self.parser.get_format_instructions(),
          'valid_examples': self.PROMPT_VALID_EXAMPLES,
          'invalid_examples': state_dict['error']
        })
    else:
      prompt = PromptTemplate(template=self.PROMPT_WITH_NO_ERROR,
        input_variables=["input"],
        partial_variables={
          'task': self.PROMPT_TASK,
          'format_instructions': self.parser.get_format_instructions(),
          'valid_examples': self.PROMPT_VALID_EXAMPLES
        })
        
    return prompt

  def _detect_p_q_items(self, state: GraphState):        
    prompt = self._get_prompt(state) 
    chain = prompt | self.llm | self.parser

    state_dict = state['keys']
    input = state_dict['input']
    iter = state_dict['iterations']

    logger.debug(prompt.format(input=input))

    return {
      'keys': {
        'input': input,
        'pq_items': chain.invoke({'input': input}),
        'iterations': iter + 1
      }
    }

  def _decide_valid_items(self, state: GraphState):
    state_dict = state['keys']
    logger.debug(state_dict['pq_items'])
    input = state_dict['input']
    detected_items = state_dict['pq_items']
    if state_dict['iterations'] < self.MAX_ATTEMPTS:
      if 'Q' not in detected_items or not detected_items['Q']:
        logger.debug('error no q items, retry it')
        state_dict['error'] = f'Input:\n{input}\nOutput:\n{json.dumps(detected_items)}\nError: There is no Q item, you should provide at least one Q item.'
        return 'detect_p_q_items'
      item_no_subs = self._item_not_in_input(detected_items, input)
      if item_no_subs:
        logger.debug('error no q items, retry it')
        state_dict['error'] = f'Input:\n{input}\nOutput:\n{json.dumps(detected_items)}\nError: The item "{item_no_subs}" is not a substring of the input.'
        return 'detect_p_q_items'
    else:
      logger.debug('no more attempts')
      state_dict['pq_items'] = None
    return 'end'

  def _apply_elements(self, text, pq_items):
    enriched_text = text
    enriched_text_ui = text
    if 'P' in pq_items:
      for p_text in pq_items['P']:
        p_number = self.wikibase.getProperty(p_text)
        if p_number:
          enriched_text = enriched_text.replace(p_text, self.ITEM_CONTEXT_TEMPLATE.substitute(text=p_text, number=p_number))
          enriched_text_ui = enriched_text_ui.replace(p_text, 
            self.ITEM_CONTEXT_UI_TEMPLATE.substitute(text=p_text, number=p_number, type='Property'))
    for q_text in pq_items['Q']:
      q_number = self.wikibase.getQItem(q_text)
      if q_number:
        enriched_text = enriched_text.replace(q_text, self.ITEM_CONTEXT_TEMPLATE.substitute(text=q_text, number=q_number))
        enriched_text_ui = enriched_text_ui.replace(q_text, 
          self.ITEM_CONTEXT_UI_TEMPLATE.substitute(text=q_text, number=q_number, type='Item'))
    return enriched_text, enriched_text_ui

  def enrich(self, text, debug = False):
    results = self.workflow.invoke({'keys': {'input': text, 'iterations': 0}}, config=self.workflow_config)
    pq_items = results['keys']['pq_items']
    if pq_items:
      return self._apply_elements(text, pq_items)
    else:
      return None, None
