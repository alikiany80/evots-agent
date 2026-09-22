from evots.bank import detect as bank_detect
SPEC = {'model': 'bottom_up', 'penalty': 11.2}

def detect(train, x, seed):
    return bank_detect(train, x, SPEC, seed)
