"""Conservative source binding for a single destination-only replacement."""
from copy import deepcopy
import re
from algorithms.source_spans import correction_candidates, source_tokens


GOAL = (r'(?:the\s+)?(?:(?!(?:next|left|right|above|below|under|behind|beside|between|near|not|no|both|all)\b)'
        r'[a-z]+\s+)?(?:goal|target|marker|region|area|disk|circle|ring|zone)')
ORIGINAL = re.compile(
    r'\s*(?:please\s+)?(?P<action>move|put|place|take|carry|send|transfer|bring|transport|deliver|relocate|set)\s+'
    r'(?P<object>(?:the\s+)?(?P<color>[a-z]+)(?:\s+(?:cube|block|piece|item|object))?)\s+'
    r'(?:to|onto|into|on|in|at)\s+(?P<destination>'+GOAL+r')\s*[.;:!,\-]?\s*', re.IGNORECASE)
REPLACEMENT = re.compile(
    r'\s*(?:please\s+)?(?:use|choose|select|pick|make that)\s+'
    r'(?P<destination>'+GOAL+r')\s*(?:instead)?[.!]?\s*', re.IGNORECASE)
OBJECT_TO_INHERITED_GOAL = re.compile(
    r'\s*(?P<action>move|put|place|take|carry|send|transfer|bring|transport|deliver|relocate|set)\s+'
    r'(?P<object>(?:the\s+)?(?P<color>[a-z]+)(?:\s+(?:cube|block|piece|item|object))?)\s+'
    r'there\s*[.;:!,\-]?\s*', re.IGNORECASE)


def bind_destination_only(intent, instruction):
    """Bind roles only when the entire source proves one task and a target-only edit.

    No scene or supported-color list is used. Unknown colors stay unknown. This
    narrow rule cannot consume extra moves, multiple objects, pronouns or a new
    moved-object clause. Unmatched language remains with the normal extractor.
    """
    tasks = intent['tasks']
    if (len(tasks) != 1 or tasks[0]['action'] != 'place' or tasks[0]['object_quantity'] != 'one'
            or tasks[0]['destination_type'] != 'goal'):
        return intent, None
    tokens = source_tokens(instruction)
    candidates = correction_candidates(instruction, tokens)
    if not candidates:
        return intent, None
    candidate = candidates[-1]
    cue_start = tokens[candidate['cue_span']['start']]['start_char']
    replacement_start = tokens[candidate['replacement_span']['start']]['start_char']
    first_cue_start = tokens[candidates[0]['cue_span']['start']]['start_char']
    original = ORIGINAL.fullmatch(instruction[:first_cue_start])
    object_offset = 0
    replacement = REPLACEMENT.fullmatch(instruction[replacement_start:])
    if original is None or replacement is None:
        return intent, None
    old_destination = original.group('destination')
    previous_colors = {original.group('color').casefold()}
    for previous, following in zip(candidates, candidates[1:]):
        start = tokens[previous['replacement_span']['start']]['start_char']
        end = tokens[following['cue_span']['start']]['start_char']
        clause = instruction[start:end]
        full_task = ORIGINAL.fullmatch(clause)
        object_edit = full_task or OBJECT_TO_INHERITED_GOAL.fullmatch(clause)
        target_edit = REPLACEMENT.fullmatch(clause)
        if object_edit:
            original, object_offset = object_edit, start
            previous_colors.add(original.group('color').casefold())
            if full_task:
                old_destination = full_task.group('destination')
        elif target_edit:
            old_destination = target_edit.group('destination')
        else:
            return intent, None
    original_color = original.group('color').casefold()
    destination = replacement.group('destination')
    task = tasks[0]
    # A hallucinated unrelated object is never repaired into an available one.
    if task['object_color'] not in previous_colors:
        quoted = task['object_evidence'].strip().casefold()
        if not quoted or quoted not in destination.casefold() or not task['object_color'] or not re.search(
                r'\b'+re.escape(task['object_color'])+r'\b', destination, re.IGNORECASE):
            return intent, None
    normalized = deepcopy(intent)
    normalized['tasks'][0].update(
        action_evidence=original.group('action'), object_color=original_color,
        object_evidence=original.group('object'), destination_type='goal', destination_color=None,
        destination_evidence=destination)
    normalized['correction_evidence'] = instruction[replacement_start:]
    positions = []
    for role, match, field, offset in [('action',original,'action',object_offset), ('object',original,'object',object_offset),
                                      ('object_color',original,'color',object_offset),
                                      ('destination',replacement,'destination',replacement_start)]:
        start, end = match.span(field)
        start, end = start+offset, end+offset
        selected = [t['index'] for t in tokens if t['start_char'] < end and t['end_char'] > start]
        positions.append(dict(field=role, start_char=start, end_char=end, text=instruction[start:end],
                              token_start=min(selected), token_end=max(selected)+1))
    return normalized, dict(passed=True, correction_scope='destination',
        reason='A complete single-task source and destination-only replacement preserve original object evidence',
        source_positions=positions, inherited_object=original.group('object'),
        replaced_destination=old_destination, active_destination=destination)
