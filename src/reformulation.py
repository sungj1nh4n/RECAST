"""Type-conditioned query reformulation (§III-B, Fig. 2)."""
import json
import re

# [PARTIALLY_SPECIFIED] Reconstructed prompts and deterministic checks (§III-B2).
ROUTE = ('Decide whether grounding abstract, spoken or non-observable cues is needed and '
         'whether a compound question or options need decomposition into visual clues. '
         'Return JSON with boolean fields need_ground and need_decompose.')
GROUND = ('Map abstract actions, intent and causes to visible evidence. Use speech as a '
          'temporal pointer to concurrent action. Exclude overlays, subtitles and audio '
          'as evidence. Preserve each option. Return only visual clues.')
DECOMPOSE = ('Separate compound clues and extract the visual core of every answer option. '
             'Keep their order. Use grounded clues where available.')
SYNTHESIZE = ('Write one declarative noun-phrase-centric visual description, at most 25 '
              'words. Include a distinctive attribute per subject and one either ... or '
              'clause covering all four options in order. Drop scene context first. '
              'Strip interrogative and meta tokens, ignore overlays and audio. Return '
              'JSON with description and option_spans, four distinct verbatim phrases '
              'from the disjunction, one per option in input order.')
BANNED = set('who what when where why which how question option options answer answers video'.split())


def check_draft(raw):
    """Deterministic, ITM-free validation (§III-B2). Spans approximate option coverage."""
    try:
        draft = json.loads(raw)
    except ValueError:
        draft = None
    if not isinstance(draft, dict) or not isinstance(draft.get('description'), str):
        return '', ['Return a JSON object containing a description and option_spans']
    desc = draft['description'].strip()
    errors = []
    if not desc or len(desc.split()) > 25:
        errors.append('Use 1 to 25 words')
    words = re.findall(r"\b[\w'-]+\b", desc.casefold())
    banned = sorted(set(words) & BANNED)
    if banned or '?' in desc:
        errors.append(f'Remove question marks and banned tokens {banned}')
    if words.count('either') != 1 or 'or' not in words[words.index('either') + 1:]:
        errors.append('Use one either ... or clause')
    spans = draft.get('option_spans')
    clause = re.split(r'\beither\b', desc.casefold())[-1]
    if isinstance(spans, list) and len(spans) == 4 and all(isinstance(s, str) for s in spans):
        spans = [s.strip().casefold() for s in spans]
        if len(set(spans)) == 4 and all(spans):
            for span in spans:
                match = re.search(r'(?<!\w)' + re.escape(span) + r'(?!\w)', clause)
                if match is None:
                    break
                clause = clause[match.end():]
            else:
                return desc, errors
    errors.append('Supply four distinct, ordered visual spans quoted from the either clause')
    return desc, errors


def reformulate(question, options, generate):
    """generate(stage, instruction, payload) returns raw model text (§IV-A5)."""
    if not isinstance(question, str) or not question.strip() or isinstance(options, str) or len(options) != 4:
        raise ValueError('Supply a question and four answer options')
    if any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError('Each option must be nonempty text')
    payload = dict(question=question, options=list(options))
    route = json.loads(generate('route', ROUTE, payload))
    for stage, prompt in [('ground', GROUND), ('decompose', DECOMPOSE)]:
        flag = 'need_' + stage
        if not isinstance(route, dict) or type(route.get(flag)) is not bool:
            raise ValueError(f'Router must provide a boolean {flag}')
        if route[flag]:
            payload[stage] = generate(stage, prompt, payload)
    for attempt in range(2):  # §III-B2 allows one bounded correction
        raw = generate('synthesize', SYNTHESIZE, payload)
        description, errors = check_draft(raw)
        if not errors:
            return description
        payload.update(previous_draft=raw, violations=errors)
    raise ValueError('Corrected description failed validation: ' + ', '.join(errors))
