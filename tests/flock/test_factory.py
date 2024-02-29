def test_create_two_tier_flock():
    from flox.flock.factory import create_two_tier_flock

    for n in [1, 10, 100]:
        flock = create_two_tier_flock(n)
        assert flock.number_of_workers == n
        assert flock.leader is not None
