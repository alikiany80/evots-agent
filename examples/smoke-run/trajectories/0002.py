from evots.bank import detect as bank_detect
SPEC = {'model': 'bottom_up'}

def detect(train, x, seed):
    return bank_detect(train, x, SPEC, seed)
