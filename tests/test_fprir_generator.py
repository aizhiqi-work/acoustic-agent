from types import SimpleNamespace

from acoustic_agent.floorplan_resource import FloorplanResource
from scripts.generate_fprir import (
    _build_jobs_for_floorplan,
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
    assert _tier_label(8000) == "adapt-8k"
    assert _tier_label(15376, 15376) == "adapt-full"


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


def test_adapt_variants_have_unique_ids_seeds_and_expected_channel_families():
    resource = FloorplanResource()
    jobs = []
    for variant in range(3):
        jobs.extend(
            _build_jobs_for_floorplan(
                resource,
                0,
                split="train",
                variant_index=variant,
                include_same_room=variant < 2,
                include_cross_room=True,
                include_distributed=False,
                max_distributed_mics=4,
                include_motion=variant == 0,
                motion_distance_m=1.0,
                motion_spacing_m=0.25,
            )
        )

    assert len({job.item_id for job in jobs}) == len(jobs)
    assert sum(job.kind == "same_room_mono" for job in jobs) == 2
    assert sum(job.kind == "cross_room_circular4" for job in jobs) == 3
    assert sum(job.kind == "moving_source_mono" for job in jobs) == 1
    assert len({job.material_seed for job in jobs}) == 3
    assert {job.variant_index for job in jobs} == {0, 1, 2}
