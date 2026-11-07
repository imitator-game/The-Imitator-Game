"""Explicit model/key-group bindings. No model or provider fallback."""
PROFILES = {
    'sol': {'model': 'gpt-6.1-sol', 'effort': 'xhigh', 'credential': 'xfast.json'},
    'astra': {'model': 'gpt-6-astra', 'effort': 'xhigh', 'credential': 'xfast.json'},
    'deepseek': {'model': 'deepseek/deepseek-v4.1-flash', 'effort': 'high',
                'credential': 'xfast_deepseek.json',
                'checkpoint_note': 'Official Flash alias maps to 1017; gateway backend checkpoint is not independently attested.'},
}


def get_profile(name):
    if name not in PROFILES:
        raise ValueError('Unknown model profile; implicit fallback is forbidden')
    return dict(PROFILES[name], name=name, provider='xfast')
