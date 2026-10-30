"""Conservative dispatch only; each skill still performs full LLM validation."""
import re


class RoutingError(ValueError):
    pass


def select_skill(instruction, requested='auto', offline=False):
    if requested not in ('auto','pick_place','push'):
        raise RoutingError('Unknown skill')
    push=bool(re.search(r'\bpush(?:ing)?\b|推',instruction,re.I))
    place=bool(re.search(r'\b(?:pick(?:ing)?|place|placing|put|grasp|stack)\b|抓|放|搬|拿',instruction,re.I))
    if push and place:
        raise RoutingError('Mixed pushing and pick-and-place sequences are not supported; use separate commands')
    if requested=='pick_place' and push:
        raise RoutingError('Instruction requests pushing but --skill is pick_place')
    skill=('push' if push else 'pick_place') if requested=='auto' else requested
    if offline and skill=='push':
        raise RoutingError('--plan-file is supported only for existing pick-and-place replay')
    return skill


def complete_log(log, control_steps):
    """Add common fields without removing legacy plan/execution fields."""
    execution=log.get('execution') or {}
    log['schema_version']=1
    log['control_steps']=control_steps
    log['physical_success']=execution.get('success')
    log['reason']=log.get('error') or execution.get('reason') or (log.get('plan') or {}).get('reason')
    return log
