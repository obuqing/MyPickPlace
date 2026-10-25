"""Source-role edits and independent-task checks; never consult scene availability."""
from copy import deepcopy
import re
from algorithms.correction_roles import ORIGINAL, OBJECT_TO_INHERITED_GOAL, bind_destination_only
from algorithms.source_spans import correction_candidates, source_tokens


NAMED_OBJECT = r'(?P<object>(?:the\s+)?(?P<color>[a-z]+)(?:\s+(?:cube|block|piece|item|object))?)'
OBJECT_EDIT = re.compile(r'\s*(?:(?:use|choose|select|pick)\s+)?'+NAMED_OBJECT+
    r'(?:\s+(?:should|must)\s+(?:go|be placed)\s+there)?\s*(?:instead)?[.!]?\s*',re.IGNORECASE)
PLACEMENT = re.compile(r'\b(?:move|put|place|take|carry|send|transfer|bring|transport|deliver|relocate|set)\s+'
                      r'(?:the\s+)?(?P<color>[a-z]+)\b',re.IGNORECASE)


def independent_task_evidence(instruction):
    tokens=source_tokens(instruction)
    candidates=correction_candidates(instruction,tokens)
    offset=tokens[candidates[-1]['replacement_span']['start']]['start_char'] if candidates else 0
    active=instruction[offset:]
    matches=[]
    for match in PLACEMENT.finditer(active):
        word=match.group('color').casefold()
        if word in {'it','that','this','them','there','a','an','both','all','any','one','something','not','no',
                    'to','into','onto','on','in','at','for','while','you','and','then','still','where'}:
            continue
        prefix=re.findall(r"[\w']+",active[:match.start()].casefold())[-2:]
        if match.group().split()[0].casefold() == 'put' and prefix and prefix[-1] in {'stay','stays','remain','remains'}:
            continue
        if any(w in {'not',"don't",'never'} for w in prefix):
            continue
        matches.append(match)
    for first,second in zip(matches,matches[1:]):
        if first.group('color').casefold()!=second.group('color').casefold() or re.search(
                r'\b(goal|target|marker|area|disk|circle|region|there)\b',active[first.end():second.start()],re.IGNORECASE):
            return [dict(start_char=m.start()+offset,end_char=m.end()+offset,text=m.group()) for m in matches]
    return []


def bind_role_correction(intent,instruction):
    normalized, audit=bind_destination_only(intent,instruction)
    if audit:
        return normalized,audit
    tasks=intent['tasks']
    if not 1 <= len(tasks) <= 2 or any(t['action']!='place' or t['object_quantity']!='one' for t in tasks):
        return intent,None
    tokens=source_tokens(instruction)
    candidates=correction_candidates(instruction,tokens)
    if len(candidates)!=1:
        return intent,None
    candidate=candidates[0]
    cue=tokens[candidate['cue_span']['start']]['start_char']
    offset=tokens[candidate['replacement_span']['start']]['start_char']
    original=ORIGINAL.fullmatch(instruction[:cue])
    if original is None:
        return intent,None
    replacement=instruction[offset:]
    both=ORIGINAL.fullmatch(replacement)
    object_edit=both or OBJECT_TO_INHERITED_GOAL.fullmatch(replacement) or OBJECT_EDIT.fullmatch(replacement)
    if object_edit is None:
        return intent,None
    # Generic target-only wording is handled only by the target binder, not here.
    color=object_edit.group('color').casefold()
    if color in {'it','that','this','them','there','a','an','both','all','any','one','not','no'}:
        return intent,None
    destination=both.group('destination') if both else original.group('destination')
    action_match=both or original
    action_offset=offset if both else 0
    normalized=deepcopy(intent)
    task=deepcopy(tasks[-1])
    task.update(action='place',action_evidence=action_match.group('action'),
                object_color=color,object_evidence=object_edit.group('object'),
                destination_type='goal',destination_color=None,destination_evidence=destination)
    normalized.update(tasks=[task],correction_evidence=replacement)
    positions=[]
    for role,match,field,start_offset in [('action',action_match,'action',action_offset),
        ('object',object_edit,'object',offset),('object_color',object_edit,'color',offset),
        ('destination',both or original,'destination',offset if both else 0)]:
        start,end=match.span(field)
        start,end=start+start_offset,end+start_offset
        selected=[t['index'] for t in tokens if t['start_char']<end and t['end_char']>start]
        positions.append(dict(field=role,start_char=start,end_char=end,text=instruction[start:end],
                              token_start=min(selected),token_end=max(selected)+1))
    return normalized,dict(passed=True,correction_scope='both' if both else 'object',
        reason='Complete source clauses bind explicit replacement roles and cancel the earlier task',
        source_positions=positions,cancelled_tasks=max(0,len(tasks)-1),
        inherited_destination=None if both else destination)
