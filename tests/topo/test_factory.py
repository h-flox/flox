def test_create_two_tier_flock():
    from flox.topos.factory import create_standard_flock

    for n in [1, 10, 100]:
        flock = create_standard_flock(n)
        assert flock.number_of_workers == n
        assert flock.leader is not None
