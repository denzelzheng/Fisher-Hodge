"""Training and test sets.  a and b are drawn uniformly in the ranges below.  A homogeneous start is the exact
well-mixed solution copied to every site (the limit-cycle test); a spatial start is a random smooth field around the
fixed point (a, b / a) (the pattern test).  The 64 x 64 set has the spacing of the training grid.

    key         label                                       count  steps  grid  a, b
    train       -                                           64     50     32    a 0.9-1.1, b 2.3-3.0
    training    training conditions                         32     200    32    a 0.9-1.1, b 2.3-3.0
    b_high      b 3.2-3.7                                   32     200    32    a 0.9-1.1, b 3.2-3.7
    b_far       b 4.0-4.6                                   32     200    32    a 0.9-1.1, b 4.0-4.6
    a_high      a 1.3-1.5, b 1.15-1.5 x (1 + a^2)           32     200    32    both chemostats outside training
    turing      Turing regime (below the Hopf line)         32     200    32    a 1.8-2.2, b 3.4-3.9
    grid64      64 x 64 grid                                16     200    64    a 0.9-1.1, b 2.3-3.0
In 'train' every even trajectory starts from a point of a well-mixed burn-in with 0.5 % noise per site and every
odd one from a random smooth field.  In the 32 x 32 test sets the first half starts homogeneous and the second half
spatial.  In the Turing set a, b lie below the Hopf line by at least 0.3 and above the Turing threshold by at least
0.3, so homogeneous starts decay and spatial starts form stationary patterns.  'grid64' has spatial starts only.
"""
import math

from .config import W_TRAIN, T_TEST, DIFF

# (count, steps, grid size, seed); the seeds fix the random streams of the stored data
GEN = dict(train=(64, W_TRAIN, 32, 1), training=(32, T_TEST, 32, 2), b_high=(32, T_TEST, 32, 3), b_far=(32, T_TEST, 32, 4),
           a_high=(32, T_TEST, 32, 5), turing=(32, T_TEST, 32, 6), grid64=(16, T_TEST, 64, 7))
TEST_SETS = ['training', 'b_high', 'b_far', 'a_high', 'turing', 'grid64']
LABELS = dict(training='training conditions', b_high='b 3.2–3.7', b_far='b 4.0–4.6',
              a_high='a 1.3–1.5 (a and b outside training)', turing='Turing regime (below the Hopf line)',
              grid64='64×64 grid')


def sample_ab(rng, kind):
    """the condition (a, b) of one trajectory of set `kind`"""
    if kind in ('train', 'training', 'grid64'):
        return rng.uniform(0.9, 1.1), rng.uniform(2.3, 3.0)
    if kind == 'b_high':
        return rng.uniform(0.9, 1.1), rng.uniform(3.2, 3.7)
    if kind == 'b_far':
        return rng.uniform(0.9, 1.1), rng.uniform(4.0, 4.6)
    if kind == 'a_high':
        a = rng.uniform(1.3, 1.5)
        return a, (1 + a * a) * rng.uniform(1.15, 1.5)
    if kind == 'turing':
        a, b = rng.uniform(1.8, 2.2), rng.uniform(3.4, 3.9)
        assert (1 + a * math.sqrt(DIFF[0] / DIFF[1])) ** 2 + 0.3 < b < 1 + a * a - 0.3
        return a, b
    raise ValueError(kind)
