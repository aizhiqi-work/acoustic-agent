from types import SimpleNamespace

from scripts.generate_fprir import _shard_index_is_complete, _tier_label


def test_shard_index_requires_every_planned_item_exactly_once():
    jobs = [SimpleNamespace(item_id="item_a"), SimpleNamespace(item_id="item_b")]

    assert _shard_index_is_complete(
        [{"group": "item_a"}, {"group": "item_b"}],
        jobs,
    )
    assert not _shard_index_is_complete([{"group": "item_a"}], jobs)
    assert not _shard_index_is_complete(
        [{"group": "item_a"}, {"group": "item_a"}],
        jobs,
    )
    assert not _shard_index_is_complete(
        [{"group": "item_a"}, {"group": "item_b"}, {"group": "item_c"}],
        jobs,
    )


def test_adapt_tier_labels_are_stable():
    assert _tier_label(10) == "adapt-10"
    assert _tier_label(1000) == "adapt-1k"
    assert _tier_label(6000) == "adapt-6k"
