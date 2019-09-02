def contains_any_no_voids(values):
    return any([value != 0 for value in values])


def rotate_matrix(matrix):
    return zip(*matrix[::-1])


def first_values_class(dictionary):
    return next(iter(dictionary.values())).__class__
