def logistic_map(x, r):
    return r*x*(1-x)


def iterate_f(it, first_x, r):

    prev_x = first_x
    ret = []

    for _ in range(it):
        new_x = logistic_map(prev_x, r)
        ret.append(new_x)
        prev_x = new_x

    return ret