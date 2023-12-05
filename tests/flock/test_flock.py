from flox.flock import Flock


def test_str_ids():
    topo = {
        "zero": {
            "kind": "leader",
            "globus_compute_endpoint": None,
            "proxystore_endpoint": None,
            "children": [1, 2],
        },
        "one": {
            "kind": "worker",
            "globus_compute_endpoint": None,
            "proxystore_endpoint": None,
            "children": [],
        },
        "two": {
            "kind": "worker",
            "globus_compute_endpoint": None,
            "proxystore_endpoint": None,
            "children": [],
        },
    }
    failure = False
    try:
        Flock.from_dict(topo)
    except Exception:
        failure = True
    assert failure
