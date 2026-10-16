# utils.py - Hybrid Text Processing for Fake News Detection
import re
import spacy

# Load spaCy model once (if available)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    USE_SPACY = True
except Exception:
    nlp = None
    USE_SPACY = False

# Enhanced stopwords list
ENHANCED_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'not', 'only', 'just', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
}

def clean_text(text: str) -> str:
    """
    Advanced unified cleaner for training and inference:
    - Lowercase and basic cleanup
    - Remove URLs, emails, special characters
    - Remove extra whitespace
    - Lemmatize using spaCy if available
    - Remove stopwords
    - Return cleaned text with meaningful words only
    """
    text = str(text).lower()

    # Remove URLs and emails
    text = re.sub(r'http\S+|www\.\S+|[\w\.-]+@[\w\.-]+\.\w+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove mentions and hashtags but keep content
    text = re.sub(r'@\w+|#', ' ', text)

    # Normalize numbers to <NUM> token (preserves sentence structure)
    # "scored in the 68th minute" → "scored in the NUM minute" (not broken gibberish)
    text = re.sub(r'\b\d+(?:[.,]\d+)*%?\b', ' NUM ', text)

    # Remove special characters and remaining digits (keep letters, spaces, hyphens, and NUM)
    text = re.sub(r"[^a-z\s\-]", " ", text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if USE_SPACY and text:
        try:
            doc = nlp(text)
            # Lemmatize and filter
            tokens = [
                tok.lemma_ for tok in doc
                if not tok.is_stop and tok.is_alpha and tok.lemma_ not in ENHANCED_STOPWORDS
            ]
            return " ".join(tokens).strip()
        except Exception:
            pass

    # Fallback: remove common stopwords manually
    tokens = [word for word in text.split() if word not in ENHANCED_STOPWORDS and len(word) > 2]
    return " ".join(tokens).strip()

def extract_features(text: str) -> dict:
    """
    Extract linguistic features useful for fake news detection:
    - sentence count
    - average sentence length
    - question count
    - exclamation count
    - all caps word count
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    total_words = len(text.split())
    avg_sent_length = total_words / len(sentences) if sentences else 0

    question_count = text.count('?')
    exclamation_count = text.count('!')
    all_caps_words = len([w for w in text.split() if w.isupper() and len(w) > 1])

    return {
        'sentence_count': len(sentences),
        'avg_sentence_length': avg_sent_length,
        'question_count': question_count,
        'exclamation_count': exclamation_count,
        'all_caps_count': all_caps_words,
        'total_words': total_words
    }

def get_conspiracy_indicators(text: str) -> int:
    """
    Count conspiracy theory language indicators
    """
    conspiracy_keywords = [
        'cover up', 'hidden truth', 'secret', 'conspiracy', 'prove', 'liar',
        'government lies', 'fake news', 'mainstream media', 'deep state',
        'shadow government', 'illuminati', 'elite', 'new world order'
    ]

    text_lower = text.lower()
    count = sum(text_lower.count(keyword) for keyword in conspiracy_keywords)
    return count

def get_sensationalism_score(text: str) -> float:
    """
    Calculate sensationalism score based on language patterns
    """
    sensational_words = [
        'shocking', 'stunning', 'exclusive', 'breaking', 'unbelievable',
        'incredible', 'amazing', 'horrible', 'tragic', 'devastating',
        'bombshell', 'scandal', 'massive', 'huge', 'enormous'
    ]

    text_lower = text.lower()
    count = sum(text_lower.count(word) for word in sensational_words)
    total_words = len(text.split())

    return (count / total_words) if total_words > 0 else 0

def _find_event_word(text: str) -> str:
    """Find the strongest event/action word in the text, if any.

    Named-entity extraction alone finds *who/where* a story is about, but
    not *what happened* — "Trump" matches any news mentioning him, while
    "Trump strikes" or "Trump Iran strikes" narrows to the actual event.
    This covers common high-signal news-event verbs/nouns across politics,
    disasters, sports, and business, so search queries can include the
    specific happening, not just the named subject.
    """
    event_words = [
        # conflict / disaster
        'strikes', 'struck', 'attacks', 'attacked', 'bombing', 'bombed',
        'crash', 'crashed', 'collision', 'explosion', 'explodes',
        'killed', 'dies', 'died', 'dead', 'death', 'earthquake', 'flood',
        'fire', 'wildfire', 'shooting', 'shot', 'war', 'invasion', 'ceasefire',
        # politics / government
        'elects', 'elected', 'election', 'resigns', 'resigned', 'resignation',
        'arrested', 'arrest', 'indicted', 'indictment', 'sentenced', 'sworn',
        'impeached', 'impeachment', 'signs', 'signed', 'vetoes', 'vetoed',
        'announces', 'announced', 'wins', 'won', 'loses', 'lost',
        # economy
        'raises', 'raised', 'cuts', 'hikes', 'inflation', 'recession',
        'bankruptcy', 'merger', 'acquires', 'acquisition', 'ipo',
        # general high-signal
        'launches', 'launched', 'recalls', 'recalled', 'banned', 'ban',
        'protest', 'protests', 'strike',
    ]
    words_lower = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    for w in event_words:
        if w in words_lower:
            return w
    return ""


def extract_keyword_phrases(text: str, max_keywords: int = 5) -> list:
    """Extract main keyword phrases from text, ranked by specificity.

    Returns a list of ranked phrases (e.g. ["Ajit Pawar", "Baramati
    Airport"]) rather than a joined string — callers that need to combine
    them with explicit OR/AND/quoting (like a search API) should use this
    instead of extract_keywords, since reverse-parsing a joined string is
    fragile when phrases themselves contain spaces.

    Groups consecutive capitalized words into full phrases (e.g. "Ajit
    Pawar", "Baramati Airport") rather than treating each word as a separate
    keyword slot — splitting names word-by-word previously caused full names
    to get truncated mid-way when several other capitalized words appeared
    earlier in the sentence (e.g. "Maharashtra Deputy Chief Minister Ajit"
    cutting off before "Pawar"). Phrases naming a person (immediately
    following a title word like "Minister"/"President") are ranked highest,
    since a newsmaker's name is usually the most useful search term; ties
    are broken by word count, not raw character length (so "Ajit Pawar"
    correctly outranks "Baramati Airport" despite being a shorter string).
    Junk words (sentence-starter "The", weekday/month names, generic
    news-filler words) are filtered out so they don't pollute the query.

    The single strongest entity is paired with an event/action word when one
    is found (e.g. "Trump" + "strikes" -> "Trump strikes"), since a bare
    name is often too broad a search term — it matches any news mentioning
    that person, not specifically the story being checked.
    """
    junk_words = {
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june', 'july',
        'august', 'september', 'october', 'november', 'december',
        'breaking', 'news', 'today', 'yesterday', 'tomorrow', 'report', 'reports',
        'according', 'said', 'says', 'also', 'after', 'before', 'during', 'all',
    }
    title_words = {
        'deputy', 'chief', 'minister', 'president', 'prime', 'secretary',
        'governor', 'senator', 'mayor', 'director', 'ceo', 'chairman',
        'spokesperson', 'official', 'leader', 'head', 'general',
    }
    event_word = _find_event_word(text)

    if USE_SPACY and text.strip():
        try:
            doc = nlp(text)
            wanted_types = {"PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "FAC"}
            # PERSON entities ranked first — same rationale as the
            # title-adjacency heuristic in the regex fallback below.
            entities = [(ent.text.strip(), ent.label_ == "PERSON")
                        for ent in doc.ents if ent.label_ in wanted_types]
            entities.sort(key=lambda e: (e[1], len(e[0].split())), reverse=True)


            seen = set()
            keywords = []
            for ent, _ in entities:
                key = ent.lower()
                if key not in seen and key not in junk_words and len(ent) > 1:
                    seen.add(key)
                    keywords.append(ent)
                if len(keywords) >= max_keywords:
                    break

            if keywords:
                if event_word and event_word not in keywords[0].lower():
                    keywords[0] = f"{keywords[0]} {event_word}"
                return keywords[:max_keywords]
        except Exception:
            pass

    # Fallback: group consecutive capitalized words into phrases, instead of
    # treating each one as a separate keyword (which truncates full names).
    words = re.findall(r"[A-Za-z]+|\d+", text)
    phrases = []
    current = []
    follows_title = False  # tracks whether `current` started right after a title word
    current_follows_title = False
    for w in words:
        is_cap = bool(re.match(r"^[A-Z][a-zA-Z]*$", w))
        is_title = w.lower() in title_words
        is_junk = w.lower() in junk_words
        if is_cap and not is_junk and not is_title:
            if not current:
                current_follows_title = follows_title
            current.append(w)
            follows_title = False
            if len(current) >= 3:
                phrases.append((" ".join(current), current_follows_title))
                current = []
        else:
            if current:
                phrases.append((" ".join(current), current_follows_title))
                current = []
            follows_title = is_title
    if current:
        phrases.append((" ".join(current), current_follows_title))

    phrases.sort(key=lambda p: (p[1], len(p[0].split())), reverse=True)
    phrases = [p[0] for p in phrases]

    if not phrases:
        generic = re.findall(r"\w{4,}", text)
        generic = [w for w in generic if w.lower() not in ENHANCED_STOPWORDS and w.lower() not in junk_words]
        phrases = generic

    seen = set()
    keywords = []
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            keywords.append(phrase)
        if len(keywords) >= max_keywords:
            break

    if keywords and event_word and event_word not in keywords[0].lower():
        keywords[0] = f"{keywords[0]} {event_word}"

    return keywords


def extract_keywords(text: str, max_keywords: int = 5) -> str:
    """Extract main keywords from text as a single space-joined string,
    capped at max_keywords total words (not phrases).

    Thin wrapper around extract_keyword_phrases() for callers that just
    want a search string (e.g. simple ML feature extraction). Callers that
    need to combine phrases with explicit search operators (OR/AND/quoting)
    should call extract_keyword_phrases() directly instead, to avoid having
    to reverse-parse a joined string back into its component phrases.
    """
    # Request more phrases than max_keywords, since a single phrase may be
    # only 1 word while another may be 3 — capping at the phrase level
    # wouldn't reliably produce up to max_keywords words. Truncate by actual
    # word count afterward instead.
    phrases = extract_keyword_phrases(text, max_keywords * 3 or 1)
    words = []
    for phrase in phrases:
        for w in phrase.split():
            if len(words) >= max_keywords:
                break
            words.append(w)
        if len(words) >= max_keywords:
            break
    return ' '.join(words)