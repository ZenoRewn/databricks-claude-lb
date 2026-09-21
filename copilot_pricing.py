"""GitHub Copilot token-price estimates, not invoices. Author: Zeno Ren.

Snapshot: https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
Checked 2026-09-21. Rates are USD per million tokens; 1 AI credit = USD 0.01.
Only explicit model IDs/decimal-separator aliases match; no family fallback.
"""
from datetime import date
import re

SOURCE_URL = 'https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing'
CHECKED_ON = '2026-09-21'
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
    'claude-haiku-4.5': _rates(1,5,.1,1.25),
    'claude-sonnet-4': _rates(3,15,.3,3.75),
    'claude-sonnet-4.6': _rates(3,15,.3,3.75),
    'claude-opus-4.7': _rates(5,25,.5,6.25),
    'claude-opus-4.8': _rates(5,25,.5,6.25),
    'claude-opus-5': _rates(5,25,.5,6.25),
    'claude-sonnet-5': _rates(2,10,.2,2.5),
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
    'grok-4.5': (200000,_rates(4,12,1)),
    'grok-4.6': (200000,_rates(4,12,1)),
}
PROMOTION_END = {m:date(2026,12,31) for m in ('gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash')}


def _normalize(name):
    return re.sub(r'(?<=\d)\.(?=\d)','-',name.strip().lower()) if isinstance(name,str) else ''


ALIASES = {_normalize(name):name for name in MODELS}
METADATA = {'source_url':SOURCE_URL,'checked_on':CHECKED_ON,'unit':'USD per 1M tokens',
            'ai_credit_usd':AI_CREDIT_USD,'scope':'process_lifetime_reported_usage',
            'basis':'copilot_published_token_rates','actual_invoice':False}


def get_pricing(model, input_tokens=0, *, at=None):
    name=ALIASES.get(_normalize(model))
    if name is None or type(input_tokens) is not int or input_tokens<0:return None
    if name in PROMOTION_END and (at or date.today())>PROMOTION_END[name]:return None
    rates=MODELS[name];tier='default'
    if name in LONG_CONTEXT and input_tokens>LONG_CONTEXT[name][0]:
        rates=LONG_CONTEXT[name][1];tier='long_context'
    return {**rates,'model':name,'tier':tier,'checked_on':CHECKED_ON,
            'valid_through':PROMOTION_END[name].isoformat() if name in PROMOTION_END else None}


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
