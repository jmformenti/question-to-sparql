import logging
from wikibaseintegrator import wbi_helpers
from wikibaseintegrator.wbi_config import config as wbi_config


logger = logging.getLogger(f'app.{__name__}')

class WikibaseConnector():
  WB_LANGUAGE = 'en'
  WB_LIMIT = 10
  WB_USER_AGENT = 'MyWikibaseBot/1.0'

  def __init__(self):
    wbi_config['USER_AGENT'] = 'MyWikibaseBot/1.0'

  def getQItem(self, name: str) -> str:
    """Returns the Q item from my wikibase."""

    data = {
      'action': 'wbsearchentities',
      'search': name,
      'type': 'item',
      'language': self.WB_LANGUAGE,
      'limit': self.WB_LIMIT
    }
    result = wbi_helpers.mediawiki_api_call_helper(data=data, allow_anonymous=True)
    if result['search']:
      return result['search'][0]['id']
    else:
      return None

  def getProperty(self, name: str) -> str:
    """Returns the property from my wikibase."""

    data = {
      'action': 'wbsearchentities',
      'search': name,
      'type': 'property',
      'language': self.WB_LANGUAGE,
      'limit': self.WB_LIMIT
    }
    result = wbi_helpers.mediawiki_api_call_helper(data=data, allow_anonymous=True)
    if result['search']:
      return result['search'][0]['id']
    else:
      return None
