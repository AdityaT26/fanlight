"""Bilingual (English + Hindi) tokenisation and TF-IDF vectorisation.

No downloads, no NLTK. The tokenizer keeps Devanagari words intact; Python's
``\\w`` does not match vowel signs (matras), so the default scikit-learn token
pattern would shred Hindi words into fragments.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Latin letters (incl. accented), digits and the Devanagari block minus the
# danda punctuation marks (U+0964, U+0965).
_TOKEN = re.compile(r"[0-9a-zÀ-ɏऀ-ॣ०-ॿ]+")

_ENGLISH_STOPWORDS = """
a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had
has have having he her here hers herself him himself his how i if in into is it its itself just
me more most my myself no nor not now of off on once only or other our ours ourselves out over
own said same say says she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves according told added
""".split()

_HINDI_STOPWORDS = """
का की के को में है हैं था थी थे से पर और यह वह इस उस एक ने भी तो ही कि जो कर किया गया गई गए
लिए साथ तक या हो होगा होगी होता होती होते रहा रही रहे अपने अपनी अपना कुछ कोई किसी जब तब
अब वे हम आप मैं उन्होंने उनके उनकी उनका इसके इसकी इसका बाद पहले द्वारा नहीं न थीं हुआ हुई हुए
कहा बताया दिया दी सकता सकती सकते वाले वाली वाला जा जाता जाती जाते करने करते करती ये वो उन इन
""".split()

STOPWORDS: frozenset[str] = frozenset(_ENGLISH_STOPWORDS) | frozenset(_HINDI_STOPWORDS)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stopwords and one-character noise removed."""
    return [token for token in _TOKEN.findall(text.lower()) if len(token) > 1 and token not in STOPWORDS]


def make_vectorizer(n_documents: int) -> TfidfVectorizer:
    """TF-IDF over unigrams and bigrams, sized to the corpus."""
    return TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,  # the tokenizer lower-cases
        ngram_range=(1, 2),
        min_df=2 if n_documents >= 50 else 1,
        max_df=0.6 if n_documents >= 50 else 1.0,
        max_features=80_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
