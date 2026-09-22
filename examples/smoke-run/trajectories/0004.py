from evots.bank import detect as bank_detect
SPEC = {'model': 'window', 'penalty': 12.0, 'representation': 'raw_squared'}

def detect(train, x, seed):
    return bank_detect(train, x, SPEC, seed)
