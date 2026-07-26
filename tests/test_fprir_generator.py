from types import SimpleNamespace

from scripts.generate_fprir import (
    _partition_floorplans,
    _shard_index_is_complete,
    _tier_label,
)


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


def test_floorplan_partitions_are_disjoint_and_reconstruct_global_order():
    selected = list(range(23))
    partitions = [
        _partition_floorplans(selected, partition_count=4, partition_rank=rank)
        for rank in range(4)
    ]

    assert sum(len(partition) for partition in partitions) == len(selected)
    assert set().union(*(set(partition) for partition in partitions)) == set(selected)
    assert all(
        set(partitions[left]).isdisjoint(partitions[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )


def test_floorplan_partition_rejects_invalid_rank():
    try:
        _partition_floorplans([1, 2, 3], partition_count=2, partition_rank=2)
    except ValueError as exc:
        assert "partition rank" in str(exc)
    else:
        raise AssertionError("invalid partition rank should fail")
