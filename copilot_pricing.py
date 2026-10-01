"""GitHub Copilot token-price estimates, not invoices. Author: Zeno Ren.

Snapshot: https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
Bundled 2026-10-01. Rates are USD per million tokens; 1 AI credit = USD 0.01.
Only explicit model IDs/decimal-separator aliases match; no family fallback.
"""
import asyncio
from datetime import date, datetime, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile

import httpx

SOURCE_URL = 'https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing'
CHECKED_ON = '2026-10-01'
AI_CREDIT_USD = 0.01


def _rates(input_price, output_price, cached_input, cache_write=0):
    return {'input':input_price,'output':output_price,'cache_read':cached_input,'cache_write':cache_write}


MODELS = {
    'gpt-5-mini': _rates(.25,2,.025),
    'gpt-5.3-codex': _rates(1.75,14,.175),
    'gpt-5.4': _rates(2.5,15,.25),
    'gpt-5.4-mini': _rates(.75,4.5,.075),
    'gpt-5.4-nano': _rates(.2,1.25,.02),
    'gpt-5.5': _rates(5,30,.5),
    'gpt-5.6-luna': _rates(.2,1.2,.02,.25),
    'gpt-5.6-sol': _rates(4,20,.4,5),
    'gpt-5.6-terra': _rates(2,12,.2,2.5),
    'gpt-6-astra': _rates(10,50,1,12.5),
    'gpt-6-luna': _rates(.1,.5,.01,.125),
    'gpt-6-sol': _rates(2,10,.2,2.5),
    'gpt-6.1-sol': _rates(2,10,.1,2.5),
    'claude-haiku-4.5': _rates(1,5,.1,1.25),
    'claude-sonnet-4': _rates(3,15,.3,3.75),
    'claude-sonnet-4.6': _rates(3,15,.3,3.75),
    'claude-opus-4.7': _rates(5,25,.5,6.25),
    'claude-opus-4.8': _rates(5,25,.5,6.25),
    'claude-opus-5': _rates(5,25,.5,6.25),
    'claude-opus-5.5': _rates(4,20,.2,5),
    'claude-sonnet-5': _rates(2,10,.2,2.5),
    'claude-sonnet-5.5': _rates(2,10,.2,2.5),
    'claude-opus-4.8-fast': _rates(10,50,1,12.5),
    'claude-fable-5': _rates(10,50,1,12.5),
    'claude-fable-5.1': _rates(10,50,.25,12.5),
    'gemini-3.5-flash': _rates(1.5,9,.15),
    'gemini-3.6-flash': _rates(.75,3.75,.075),
    'gemini-3.7-flash': _rates(.75,3.75,.075),
    'gemini-3.8-flash': _rates(.75,3.75,.075),
    'mai-code-1.1-flash': _rates(.2,1.2,.02),
    'grok-4.5': _rates(2,6,.5),
    'grok-4.6': _rates(2,6,.5),
    'grok-4.7': _rates(2,6,.5),
    'kimi-k2.7-code': _rates(.95,4,.19),
    'kimi-k3': _rates(3,15,.3),
}
LONG_CONTEXT = {
    'gpt-5.4': (272000,_rates(5,22.5,.5)),
    'gpt-5.5': (272000,_rates(10,45,1)),
    'gpt-5.6-luna': (200000,_rates(.4,1.8,.04,.5)),
    'gpt-5.6-sol': (272000,_rates(8,30,.8,10)),
    'gpt-5.6-terra': (272000,_rates(4,18,.4,5)),
    'gpt-6-astra': (272000,_rates(20,75,2,25)),
    'gpt-6-luna': (272000,_rates(.2,.75,.02,.25)),
    'gpt-6-sol': (272000,_rates(4,15,.4,5)),
    'gpt-6.1-sol': (272000,_rates(4,15,.2,5)),
    'grok-4.5': (200000,_rates(4,12,1)),
    'grok-4.6': (200000,_rates(4,12,1)),
    'grok-4.7': (200000,_rates(4,12,1)),
}
PROMOTION_END = {m:date(2026,12,31) for m in ('gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash')}


def _normalize(name):
    return re.sub(r'(?<=\d)\.(?=\d)','-',name.strip().lower()) if isinstance(name,str) else ''


ALIASES = {_normalize(name):name for name in MODELS}
METADATA = {'source_url':SOURCE_URL,'checked_on':CHECKED_ON,'unit':'USD per 1M tokens',
            'ai_credit_usd':AI_CREDIT_USD,'scope':'process_lifetime_reported_usage',
            'basis':'copilot_published_token_rates','actual_invoice':False}
_active = None
_refresh = None


def get_pricing(model, input_tokens=0, *, at=None):
    catalog = _active
    models = catalog['models'] if catalog else MODELS
    aliases = catalog['aliases'] if catalog else ALIASES
    long_context = catalog['long_context'] if catalog else LONG_CONTEXT
    promotion_end = catalog['promotion_end'] if catalog else PROMOTION_END
    name=aliases.get(_normalize(model))
    if name is None or type(input_tokens) is not int or input_tokens<0:return None
    if name in promotion_end and (at or date.today())>promotion_end[name]:return None
    rates=models[name];tier='default'
    if name in long_context and input_tokens>long_context[name][0]:
        rates=long_context[name][1];tier='long_context'
    return {**rates,'model':name,'tier':tier,'checked_on':catalog['checked_on'] if catalog else CHECKED_ON,
            'valid_through':promotion_end[name].isoformat() if name in promotion_end else None}


def estimate_request(model,input_tokens,output_tokens,cache_creation_tokens=0,cache_read_tokens=0,*,at=None):
    counts=(input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens)
    if not all(type(v) is int and 0<=v<2**63 for v in counts):return None
    # Copilot's Chat/Responses input is total input, including cache details.
    uncached=input_tokens-cache_creation_tokens-cache_read_tokens
    if uncached<0:return None
    pricing=get_pricing(model,input_tokens,at=at)
    if pricing is None:return None
    # N/A cache-write pricing means no separate write tariff, not free input.
    write_rate=pricing['cache_write'] or pricing['input']
    cost=(uncached*pricing['input']+output_tokens*pricing['output']+
          cache_creation_tokens*write_rate+cache_read_tokens*pricing['cache_read'])/1000000
    return {**pricing,'estimated_cost_usd':cost,'estimated_ai_credits':cost/AI_CREDIT_USD}


def estimate_history_reference(model, stats, *, at=None):
    """Current-rate reference for daily inclusive-input totals, not a past invoice.

    A day can contain short and long requests. Without request boundaries, return
    conservative component-wise price bounds rather than tiering the whole day.
    None means the model is absent; an expired/invalid known model stays unknown.
    """
    aliases = _active['aliases'] if _active else ALIASES
    if _normalize(model) not in aliases:
        return None
    unknown = {'pricing_status': 'unknown', 'estimated_cost_usd': None,
               'estimated_cost_min_usd': None, 'estimated_cost_max_usd': None,
               'pricing_basis': 'current_copilot_reference', 'pricing_source_url': SOURCE_URL}
    counts = tuple(stats.get(k, 0) for k in ('input_tokens', 'output_tokens', 'cache_creation_tokens', 'cache_read_tokens'))
    requests = stats.get('requests', 0)
    if (not {'input_tokens', 'output_tokens'}.issubset(stats)
            or not all(type(v) is int and 0 <= v < 2**63 for v in (*counts, requests))
            or (not requests and any(counts)) or counts[2] + counts[3] > counts[0]):
        return {**unknown, 'pricing_reason': 'invalid_usage_totals'}
    default = get_pricing(model, 0, at=at)
    if default is None:
        return {**unknown, 'pricing_reason': 'price_expired'}
    aggregate = get_pricing(model, counts[0], at=at)
    uncertain = requests > 1 and aggregate['tier'] != 'default'
    choices = [default, aggregate] if uncertain else [aggregate]
    # N/A cache write is billed as normal input, including when bounding tiers.
    rates = [{**r, 'cache_write': r['cache_write'] or r['input']} for r in choices]
    weights = dict(zip(('input', 'output', 'cache_write', 'cache_read'),
                       (counts[0] - counts[2] - counts[3], *counts[1:])))
    lower = sum(v * min(r[k] for r in rates) for k, v in weights.items()) / 1000000
    upper = sum(v * max(r[k] for r in rates) for k, v in weights.items()) / 1000000
    return {**unknown, 'pricing_status': 'complete',
            'estimated_cost_usd': lower if lower == upper else None,
            'estimated_cost_min_usd': lower, 'estimated_cost_max_usd': upper,
            'estimate_kind': 'range' if lower != upper else 'point',
            'pricing_checked_on': default['checked_on'],
            'pricing_reason': 'long_context_distribution_unknown' if uncertain else 'known_tier'}


def record_estimate(stats,model,input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,usage_fields=None):
    tracker=stats.setdefault('_copilot_cost',{'priced_requests':0,'subtotal_usd':0.0,'tiers':{}})
    if usage_fields is not None and not {'input_tokens','output_tokens'}.issubset(usage_fields):return
    quote=estimate_request(model,input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens)
    if quote is None:return
    tracker['priced_requests']+=1
    tracker['subtotal_usd']+=quote['estimated_cost_usd']
    tracker['tiers'][quote['tier']]=tracker['tiers'].get(quote['tier'],0)+1


def cost_view(stats):
    result={k:v for k,v in stats.items() if k!='_copilot_cost'}
    tracker=stats.get('_copilot_cost',{})
    priced=tracker.get('priced_requests',0);unpriced=max(0,stats.get('requests',0)-priced)
    subtotal=tracker.get('subtotal_usd',0.0)
    complete=unpriced==0
    result.update(priced_requests=priced,unpriced_requests=unpriced,
                  pricing_status='complete' if complete else 'partial' if priced else 'unknown',
                  estimated_cost_usd=round(subtotal,10) if complete else None,
                  estimated_ai_credits=round(subtotal/AI_CREDIT_USD,8) if complete else None,
                  known_cost_subtotal_usd=round(subtotal,10),
                  pricing_tiers=dict(tracker.get('tiers',{})))
    return result


def summarize_costs(models):
    rows=list(models);priced=sum(s['priced_requests'] for s in rows);unpriced=sum(s['unpriced_requests'] for s in rows)
    subtotal=sum(s['known_cost_subtotal_usd'] for s in rows)
    return {'estimated_total_cost_usd':round(subtotal,10) if not unpriced else None,
            'estimated_ai_credits':round(subtotal/AI_CREDIT_USD,8) if not unpriced else None,
            'known_cost_subtotal_usd':round(subtotal,10),'priced_requests':priced,'unpriced_requests':unpriced,
            'pricing_status':'complete' if not unpriced else 'partial' if priced else 'unknown'}


# Price retrieval is independent of inference clients, credentials and usage storage.
WEEK = timedelta(days=7)
RETRY = timedelta(hours=1)
MAX_PRICING_BYTES = 2 * 1024 * 1024
_log = logging.getLogger(__name__)


class _PricePage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.notes, self.text = [], {}, []
        self.table = self.row = self.cell = self.note = None
        self.suppressed = self.sup = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style'):
            self.suppressed += 1
        if self.suppressed:
            return
        if tag == 'table':
            if self.table is not None:
                raise ValueError('Nested pricing table')
            self.table = {'provider': attrs.get('aria-labelledby'), 'rows': []}
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            if attrs.get('rowspan', '1') != '1' or attrs.get('colspan', '1') != '1':
                raise ValueError('Unsupported merged pricing cell')
            self.cell = {'text': '', 'refs': [], 'has_footnote': False}
        elif tag == 'sup':
            self.sup += 1
            if self.cell is not None:
                self.cell['has_footnote'] = True
        elif tag == 'a' and self.cell is not None and 'data-footnote-ref' in attrs:
            self.cell['refs'].append(attrs.get('href', '').lstrip('#'))
        elif tag == 'li' and attrs.get('id', '').startswith('user-content-fn-'):
            self.note = attrs['id']
            self.notes[self.note] = ''

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.suppressed = max(0, self.suppressed - 1)
            return
        if self.suppressed:
            return
        if tag in ('td', 'th') and self.cell is not None:
            if self.cell['has_footnote'] and not self.cell['refs']:
                raise ValueError('Unrecognized price annotation')
            self.cell['text'] = ' '.join(self.cell['text'].split())
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table['rows'].append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None
        elif tag == 'sup':
            self.sup = max(0, self.sup - 1)
        elif tag == 'li':
            self.note = None

    def handle_data(self, text):
        if self.suppressed:
            return
        self.text.append(text)
        if self.cell is not None and not self.sup:
            self.cell['text'] += text
        if self.note is not None:
            self.notes[self.note] += text


def _model_id(display):
    name = display.replace(' (fast mode) (preview)', '-fast').lower().replace(' ', '-')
    if not re.fullmatch(r'[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*', name):
        raise ValueError('Unsupported model label')
    return name


def _price(text, *, optional=False):
    if optional and text == 'Not applicable':
        return 0.0
    if not re.fullmatch(r'\$\d+(?:\.\d+)?', text):
        raise ValueError('Invalid token price')
    value = float(text[1:])
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid token price')
    return value


def _promotion(note, name, rates):
    # Recognize the complete published rule, not just a date somewhere in a footnote.
    note = ' '.join(re.sub(r'↩\d*', '', note).split())
    match = re.fullmatch(
        r'(.+) are available at the promotional pricing of (\$[\d.]+) per 1M input tokens, '
        r'(\$[\d.]+) per 1M cached input tokens, and (\$[\d.]+) per 1M output tokens '
        r'through ([A-Za-z]+ \d{1,2}, \d{4})\.', note)
    if not match:
        raise ValueError('Unrecognized pricing footnote')
    models, inp, cached, output, end = match.groups()
    names = {_model_id(x.strip()) for x in models.replace(', and ', ', ').replace(' and ', ', ').split(',')}
    if name not in names or (rates['input'], rates['cache_read'], rates['output']) != tuple(map(_price, (inp, cached, output))):
        raise ValueError('Promotion does not match price row')
    return datetime.strptime(end, '%B %d, %Y').date()


def parse_pricing_html(html, checked_on):
    """Parse a complete validated catalog; fail closed on unfamiliar billing rules."""
    date.fromisoformat(checked_on)
    if not isinstance(html, str) or len(html.encode('utf-8')) > MAX_PRICING_BYTES:
        raise ValueError('Oversized pricing page')
    page = _PricePage()
    page.feed(html)
    page.close()
    text = ' '.join(' '.join(page.text).split())
    if ('All prices are per 1 million tokens .' not in text
            and 'All prices are per 1 million tokens.' not in text):
        raise ValueError('Unknown price unit')
    if '1 AI credit = $0.01 USD' not in text or page.table is not None:
        raise ValueError('Incomplete pricing page or unknown credit conversion')
    models, tiers, promotions, thresholds, providers = {}, {}, {}, {}, set()
    allowed = {'Model', 'Release status', 'Category', 'Tier', 'Threshold (input tokens)',
               'Input', 'Cached input', 'Cache write', 'Output'}
    for table in page.tables:
        rows = table['rows']
        if not rows:
            continue
        headers = [cell['text'] for cell in rows[0]]
        if 'Model' not in headers:
            continue
        if (set(headers) - allowed or len(set(headers)) != len(headers)
                or not {'Model', 'Input', 'Cached input', 'Output'}.issubset(headers)
                or ('Tier' in headers) != ('Threshold (input tokens)' in headers)):
            raise ValueError('Unsupported pricing columns')
        if not rows[1:]:
            continue
        provider = table['provider']
        if not provider or provider in providers:
            raise ValueError('Missing or duplicate provider')
        providers.add(provider)
        for cells in rows[1:]:
            if len(cells) != len(headers):
                raise ValueError('Incomplete price row')
            row = {key: cell['text'] for key, cell in zip(headers, cells)}
            name = _model_id(row['Model'])
            rates = _rates(_price(row['Input']), _price(row['Output']), _price(row['Cached input']),
                           _price(row.get('Cache write', 'Not applicable'), optional=True))
            tier = row.get('Tier', 'Default')
            threshold = row.get('Threshold (input tokens)', 'Not applicable')
            if tier not in ('Default', 'Long context'):
                raise ValueError('Unsupported pricing tier')
            if tier == 'Default':
                if name in models:
                    raise ValueError('Duplicate model')
                models[name] = rates
                if threshold != 'Not applicable':
                    match = re.fullmatch(r'≤\s*(\d+)K', threshold)
                    if not match:
                        raise ValueError('Unsupported default threshold')
                    thresholds[name] = int(match[1]) * 1000
            else:
                match = re.fullmatch(r'>\s*(\d+)K', threshold)
                if not match or name in tiers:
                    raise ValueError('Invalid long context row')
                tiers[name] = (int(match[1]) * 1000, rates)
            refs = [ref for cell in cells for ref in cell['refs']]
            if refs:
                ends = {_promotion(page.notes.get(ref, ''), name, rates) for ref in refs}
                if len(ends) != 1 or (name in promotions and promotions[name] not in ends):
                    raise ValueError('Conflicting promotions')
                promotions[name] = ends.pop()
    if not {'openai', 'anthropic', 'google', 'microsoft', 'xai', 'moonshot-ai'}.issubset(providers):
        raise ValueError('Missing provider tables')
    if not models or set(thresholds) != set(tiers) or any(thresholds[n] != tiers[n][0] or thresholds[n] <= 0 for n in tiers):
        raise ValueError('Incomplete long context tiers')
    aliases = {_normalize(name): name for name in models}
    if len(aliases) != len(models):
        raise ValueError('Ambiguous model aliases')
    return {'models': models, 'long_context': tiers, 'promotion_end': promotions,
            'aliases': aliases, 'checked_on': checked_on,
            'source_sha256': hashlib.sha256(html.encode('utf-8')).hexdigest()}


def install_catalog(catalog):
    global _active
    # One pointer swap: a quote always uses one complete catalog, never mixed tiers.
    _active = catalog


def get_metadata():
    catalog = _active
    return {**METADATA, 'checked_on': catalog['checked_on'] if catalog else CHECKED_ON,
            'model_count': len(catalog['models'] if catalog else MODELS),
            'source_sha256': catalog['source_sha256'] if catalog else None,
            'catalog_origin': 'refreshed' if catalog else 'bundled',
            'refresh': _refresh.status() if _refresh else {'enabled': False, 'status': 'not_started'}}


def _save_cache(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.copilot-prices-', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_cache(path):
    with path.open('rb') as stream:
        raw = stream.read(MAX_PRICING_BYTES * 2 + 4097)
    if len(raw) > MAX_PRICING_BYTES * 2 + 4096:
        raise ValueError('Oversized pricing cache')
    return json.loads(raw)


async def _disk_io(func, *args):
    task = asyncio.create_task(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Do not leave a writer running after lifespan cleanup.
        try:
            await task
        except Exception:
            pass
        raise


class PricingRefresh:
    def __init__(self, cache_path, *, enabled=True, clock=None):
        global _refresh
        self.cache_path = Path(cache_path)
        self.enabled = enabled
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.next_due = self.clock()
        self.last_success = self.last_attempt = None
        self.last_error = None
        _refresh = self

    def status(self):
        checked = (_active or {}).get('checked_on', CHECKED_ON)
        basis = self.last_success or datetime.combine(date.fromisoformat(checked), datetime.min.time(), timezone.utc)
        return {'enabled': self.enabled, 'interval_days': 7,
                'status': 'disabled' if not self.enabled else 'error' if self.last_error else 'ok' if self.last_success else 'pending',
                'last_success_at': self.last_success.isoformat() if self.last_success else None,
                'last_attempt_at': self.last_attempt.isoformat() if self.last_attempt else None,
                'next_refresh_at': self.next_due.isoformat() if self.enabled else None,
                'stale': self.clock() - basis >= WEEK, 'last_error': self.last_error}

    async def load_cache(self):
        try:
            payload = await _disk_io(_read_cache, self.cache_path)
            fetched = datetime.fromisoformat(payload['fetched_at'])
            if (payload['source_url'] != SOURCE_URL or payload['schema_version'] != 1
                    or fetched.utcoffset() != timedelta(0) or fetched > self.clock()
                    or fetched.date() < date.fromisoformat(CHECKED_ON)):
                raise ValueError('Invalid cache provenance')
            catalog = parse_pricing_html(payload['html'], fetched.date().isoformat())
            if catalog['source_sha256'] != payload['source_sha256']:
                raise ValueError('Pricing cache checksum mismatch')
            install_catalog(catalog)
            self.last_success = fetched
            self.next_due = fetched + WEEK
            self.last_error = None
        except FileNotFoundError:
            pass
        except Exception as exc:
            self.last_error = 'cache_' + type(exc).__name__
            _log.warning('Copilot price cache unavailable (%s); retaining bundled prices', type(exc).__name__)

    async def refresh_once(self, client):
        self.last_attempt = self.clock()
        try:
            async with asyncio.timeout(45):
                async with client.stream('GET', SOURCE_URL, follow_redirects=False,
                                         headers={'Accept': 'text/html'}, timeout=20) as response:
                    response.raise_for_status()
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > MAX_PRICING_BYTES:
                            raise ValueError('Oversized pricing response')
            html = raw.decode('utf-8')
            fetched = self.clock()
            catalog = parse_pricing_html(html, fetched.date().isoformat())
            payload = {'schema_version': 1, 'source_url': SOURCE_URL, 'fetched_at': fetched.isoformat(),
                       'source_sha256': catalog['source_sha256'], 'html': html}
            await _disk_io(_save_cache, self.cache_path, payload)
            install_catalog(catalog)
            self.last_success, self.next_due, self.last_error = fetched, fetched + WEEK, None
            _log.info('Copilot prices refreshed: %d models, checked %s', len(catalog['models']), catalog['checked_on'])
            return True
        except Exception as exc:
            self.last_error = type(exc).__name__
            self.next_due = self.clock() + RETRY
            _log.warning('Copilot price refresh failed (%s); retaining previous prices', self.last_error)
            return False

    async def run(self):
        await self.load_cache()
        if not self.enabled:
            return
        async with httpx.AsyncClient() as client:
            while True:
                await asyncio.sleep(max(0, (self.next_due - self.clock()).total_seconds()))
                await self.refresh_once(client)
