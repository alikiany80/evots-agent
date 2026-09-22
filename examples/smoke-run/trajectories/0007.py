from evots.bank import detect as bank_detect
SPEC = {'model': 'ensemble', 'members': [{'model': 'pelt'}, {'model': 'pelt', 'penalty': 11.2}], 'votes': 1, 'vote_tolerance': 5}

def detect(train, x, seed):
    return bank_detect(train, x, SPEC, seed)
