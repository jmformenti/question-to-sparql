import logging
import spacy
from prettytable import PrettyTable
from service.wikibase_connector import WikibaseConnector


logger = logging.getLogger(f'app.{__name__}')

class QuestionEnricher:
  TYPE_PROPERTY = 'PROPERTY'
  TYPE_ENTITY = 'ENTITY'

  def __init__(self):
    self.nlp = spacy.load("en_core_web_sm")
    self.wikibase = WikibaseConnector()

  def _insertStr(self, original, substr, position):
    return f"{original[:position]}{substr}{original[position:]}", len(substr)

  def _normalize(self, original):
    return original.replace("'s", "")

  def _apply_semantic_info_at_end(self, element, semantic_info, enriched_text, element_start_pos):
    pos = element_start_pos + len(element.text)
    return self._insertStr(enriched_text, f' ({semantic_info})', pos)

  def _apply_semantic_info_enclosed_element(self, element, element_type, semantic_info, enriched_text, element_start_pos):
    acc_num_new_chars = element_start_pos
    link_item_type = 'Property' if element_type == self.TYPE_PROPERTY else 'Item'
    link_open_tag = f'<a target="_blank" href="https://www.wikidata.org/wiki/{link_item_type}:{semantic_info}">'
    link_close_tag = f'</a> ({semantic_info})'
    
    enriched_text, num_new_chars = self._insertStr(enriched_text, link_open_tag, element_start_pos)
    acc_num_new_chars += num_new_chars
    pos_end = acc_num_new_chars + len(element.text)
    enriched_text, num_new_chars = self._insertStr(enriched_text, link_close_tag, pos_end)
    acc_num_new_chars += num_new_chars

    return enriched_text, acc_num_new_chars  

  def _apply_elements(self, text, elements):
    enriched_text = text
    enriched_text_ui = text
    acc_num_new_chars = 0
    acc_num_new_chars_ui = 0

    sorted_elements_by_pos = dict(sorted(elements.items()))

    for key, item in sorted_elements_by_pos.items():
      element_type = item['type']
      element = item['span']
      
      logger.debug(f'Applying element {element} with type {element_type}')
      semantic_info = ''
      if element_type == self.TYPE_PROPERTY:
        if element.lemma_ == 'name':
          semantic_info = 'label'
        elif hasattr(element, 'root') and element.root.lemma_ == 'instance':
          semantic_info = self.wikibase.getProperty('instance of')
        else:
          semantic_info = self.wikibase.getProperty(element.text)
          if not semantic_info and hasattr(element, 'root'):
            semantic_info = self.wikibase.getProperty(element.root.text)
      elif element_type == self.TYPE_ENTITY:
        semantic_info = self.wikibase.getQItem(self._normalize(element.text))
        if not semantic_info and hasattr(element, 'root'):
          semantic_info = self.wikibase.getQItem(element.root.text)
      else:
        raise ValueError(f"'{item.type}' not supported.")
      
      if semantic_info:
        enriched_text, num_new_chars = self._apply_semantic_info_at_end(element, semantic_info, enriched_text, key + acc_num_new_chars)
        acc_num_new_chars += num_new_chars

        enriched_text_ui, num_new_chars = self._apply_semantic_info_enclosed_element(element, element_type, semantic_info, enriched_text_ui, key + acc_num_new_chars_ui)
        acc_num_new_chars_ui += num_new_chars

    return enriched_text, enriched_text_ui

  def _remove_det(self, span):
      if not isinstance(span, spacy.tokens.token.Token) and len(span) > 0 and span[0].pos_ == 'DET':
          return self._remove_det(span[1:])
      return span

  def _remove_tokens_from(self, span, original_tokens):
    return [i for i in original_tokens if i not in range(span.start, span.end)]

  def _are_tokens_available(self, span, free_tokens):
    return all(i in free_tokens for i in range(span.start, span.end))

  def _print_debug_table_info(self, title, table):
    logger.info(f'\n## {title}')
    pretty_tab = PrettyTable(table[0])
    pretty_tab.add_rows(table[1:])
    logger.info(pretty_tab)

  def _print_debug_info(self, doc):
    tokens_table = []
    tokens_table.append(['TEXT', 'LEMMA', 'POS', 'TAG', 'DEP', 'SHAPE', 'ALPHA', 'STOP', 'HEAD TEXT', 'CHILDREN'])
    for token in doc:
      tokens_table.append([token.text, token.lemma_, f'{token.pos_} ({spacy.explain(token.pos_)})', f'{token.tag_} ({spacy.explain(token.tag_)})', f'{token.dep_} ({spacy.explain(token.dep_)})', token.shape_, token.is_alpha, token.is_stop, token.head.text, str([child for child in token.children])])
    self._print_debug_table_info('Tokens', tokens_table)

    chunks_table = []
    chunks_table.append(['TEXT', 'ROOT.TEXT', 'ROOT.DEP_', 'ROOT.HEAD.TEXT'])
    for chunk in doc.noun_chunks:
      chunks_table.append([chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text])
    self._print_debug_table_info('Noun phrases', chunks_table)

    entities_table = []
    entities_table.append(['TEXT', 'START_CHAR', 'END_CHAR', 'LABEL', 'DESCRIPTION'])
    for ent in doc.ents:
      entities_table.append([ent.text, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_)])
    self._print_debug_table_info('Entities', entities_table)

  def _find_related_dobj_token(self, token):
    token_head = token.head
    if token_head.dep_ == 'dobj':
      return token_head
    elif token_head.dep_ == 'ROOT':
      return None
    else:
      return self._find_related_dobj_token(token_head)

  def _find_nounchunk_for_token(self, token, doc):
    for nc in doc.noun_chunks:
      if token.i >= nc.start and token.i < nc.end:
        return nc
    return None

  def _create_new_element(self, span, type):
    span_with_no_det = self._remove_det(span)
    return { 'span': span_with_no_det, 'type': type}

  def _update_elements(self, new_element, elements, free_tokens, start_pos = None):
    span = new_element['span']
    if isinstance(span, spacy.tokens.token.Token):
      elements[span.idx] = new_element
      free_tokens.remove(span.i)
    else:
      if start_pos:
        elements[start_pos] = new_element
      else:
        elements[span.start_char] = new_element
      free_tokens = self._remove_tokens_from(span, free_tokens)
    return elements, free_tokens

  def _add_element(self, span, type, elements, free_tokens, start_pos = None):
    new_element = self._create_new_element(span, type)
    return self._update_elements(new_element, elements, free_tokens, start_pos)

  def enrich(self, text, debug):
    doc = self.nlp(text)

    if debug:
      self._print_debug_info(doc)

    elements = {}
    free_tokens = list(range(len(doc)))

    for chunk in doc.noun_chunks:
      if chunk.root.dep_ == 'nsubj':
        elements, free_tokens = self._add_element(chunk, self.TYPE_PROPERTY, elements, free_tokens)

    for ent in doc.ents:
      if self._are_tokens_available(ent, free_tokens):
        new_element = self._create_new_element(ent, self.TYPE_ENTITY)
        start_pos = new_element['span'].start_char - new_element['span'].sent.start_char
        elements, free_tokens = self._update_elements(new_element, elements, free_tokens, start_pos=start_pos)

    for chunk in doc.noun_chunks:
      if self._are_tokens_available(chunk, free_tokens):
        if chunk.root.dep_ == 'pobj':
          related_dobj_token = self._find_related_dobj_token(chunk.root)
          if related_dobj_token:
            self._add_element(chunk, self.TYPE_ENTITY, elements, free_tokens)
            chunk_related_dobj = self._find_nounchunk_for_token(related_dobj_token, doc)
            if chunk_related_dobj:
              elements, free_tokens = self._add_element(chunk_related_dobj, self.TYPE_PROPERTY, elements, free_tokens)
            else:
              elements, free_tokens = self._add_element(related_dobj_token, self.TYPE_PROPERTY, elements, free_tokens)
            
    for chunk in doc.noun_chunks:
      if self._are_tokens_available(chunk, free_tokens):
        if chunk.root.dep_ == 'dobj':
          elements, free_tokens = self._add_element(chunk, self.TYPE_ENTITY, elements, free_tokens)

    for token in doc:
      if token.i in free_tokens:
        if token.pos_ == 'NOUN':
          elements, free_tokens = self._add_element(token, self.TYPE_ENTITY if token.dep_ == 'attr' else self.TYPE_PROPERTY, elements, free_tokens)

    return self._apply_elements(text, elements)
