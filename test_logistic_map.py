import pytest
from math import isclose
import numpy.testing as np_testing
import numpy.random as random

from logistic import logistic_map, iterate_f


@pytest.mark.parametrize("x, r, expected", 
                         [(0.1, 2.2, 0.198), 
                          (0.2, 3.4, 0.544), 
                          (0.75, 1.7, 0.31875)])
def test_logistic_map(x, r, expected):
    assert isclose(logistic_map(x, r), expected)


@pytest.mark.parametrize("x, r, it, expected", 
                         [(0.1, 2.2, 1, [0.198]), 
                          (0.2, 3.4, 4, [0.544, 0.843418, 0.449019, 0.841163]), 
                          (0.75, 1.7, 2, [0.31875, 0.369152])])
def test_logistic_map_it(x, r, it, expected):
    np_testing.assert_array_almost_equal(iterate_f(it, x, r), expected)


def test_random_starting_point(random_state):

    xs = [random_state.random_sample() for _ in range(20)]
    r = 1.5
    it = 100
    ret_lst=[]

    for x_chosen in xs:
        last_x = iterate_f(it, x_chosen, r)[-1]
        ret_lst.append(last_x)

    print(ret_lst)
    assert all([isclose(0.32, ret_lst[i], rel_tol=0.1) for i in range(len(ret_lst))])

    

def test_chaos():

    x = 0.1
    r = 3.8
    it = 100000
    results = iterate_f(it, x, r)
    last1000 = results[-1000:]

    found_same = False
    for i, x in enumerate(last1000):
        for j, y in enumerate(last1000):
            if isclose(x, y) and not i==j:
                found_same = True

    assert all(x>0 for x in results) and all(x<1 for x in results) and not found_same

