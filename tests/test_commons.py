def test_random_generator():
    import numpy as np
    
    from flight.commons import random_generator

    rng1 = np.random.default_rng(1)
    rng2 = random_generator(1)
    assert rng1.uniform() == rng2.uniform()

    rng1 = random_generator(1)
    rng2 = random_generator(1)
    assert rng1.uniform() == rng2.uniform()
    assert rng1 is not rng2

    rng1 = random_generator(1)
    rng2 = random_generator(2)
    assert rng1.uniform() != rng2.uniform()
    assert rng1 is not rng2
    
