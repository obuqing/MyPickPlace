"""Success-gated distance curriculum bookkeeping for the carry skill."""

STAGES = [.05, .10, .15, None]


def promotion_streak(successes, attempted, streak):
    return streak + 1 if attempted > 0 and successes / attempted >= .8 else 0


def resume_cursor(summary, history):
    stage = int(summary['final_stage'])
    last = history[-1]
    if not 0 <= stage < len(STAGES) or last['step'] != summary['steps']:
        raise ValueError('Resume metadata does not match final checkpoint')
    streak = int(last['consecutive_passes']) if last['stage'] == stage else 0
    return int(summary['steps']), stage, streak
