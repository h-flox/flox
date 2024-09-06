def test_create_two_tier_flock():
    from flox.federation.topologies.factory import two_tier_topology

    for n in [1, 10, 100]:
        flock = two_tier_topology(n)
        assert flock.number_of_workers == n
        assert flock.coordinator is not None
