
import math
import re


def hitting_time_prob(time, prob_towards_goal, distance=1):
    """Probability of an asymmetric random walk hitting the point `dist` away from the origin in `time` steps.
    
    See here: https://math.stackexchange.com/questions/2794960/distribution-of-stopping-time-for-possibly-biased-random-walk"""
    assert 0 < prob_towards_goal < 1
    assert distance > 0
    n = time
    p = prob_towards_goal
    a = distance
    if n >= a and (n + a) % 2 == 0:
        return a / n * math.comb(n, (n - a) // 2) * (p)**((n + a)/2) * (1 - p)**((n - a)/2)
    else:
        return 0

def asymptotic_hitting_time(time, prob_towards_goal, distance=1):
    """Asymptotic probability of an asymmetric random walk hitting the point `dist` away from the origin in `time` steps.

    Only works for distance = 1.
    
    See here: https://math.stackexchange.com/questions/2794960/distribution-of-stopping-time-for-possibly-biased-random-walk"""
    assert 0 < prob_towards_goal < 1
    assert distance == 1
    n = time
    p = prob_towards_goal
    a = distance
    if n == 1:
        return math.nan
    return (2 * math.sqrt(p * (1 - p)))**n * math.sqrt(p) / (2 * math.sqrt(math.pi * (1 - p)) * (n - 1)**(3/2))

def parse_output(output_str):
    bounds = []
    for line in output_str.strip().split('\n'):
        bound = re.search(r'p\(\d+\) ∈ \[([\d.e-]+), ([\d.e-]+)\]', line)
        if bound:
            bounds.append((float(bound.group(1)), float(bound.group(2))))
        bound = re.search(r'p\(\d+\) = ([\d.e-]+)', line)
        if bound:
            bounds.append((float(bound.group(1)), float(bound.group(1))))
    return bounds
