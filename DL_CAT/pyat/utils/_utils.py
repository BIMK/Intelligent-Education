def make_hot_vector(indices, num_dim):
    """

    Args:
        indices: list of indices indicating 1s
        num_dim: total length of the vector

    Returns:
        v: list representing hot vector
    """

    v = [0] * num_dim
    for i in indices:
        v[i] = 1
    return v
