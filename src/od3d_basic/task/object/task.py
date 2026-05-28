from __future__ import annotations

from typing import Tuple

from od3d_basic.task.task import OD3D_Task, register_task
from od3d_basic.data.datatypes.object import ObjectPairBatch
from od3d_basic.task.datatypes.object_pair_quant import ObjectPairQuantBatch
from od3d_basic.task.datatypes.object_pair_qualit import ObjectPairQualitBatch


@register_task("ObjectTask")
class ObjectTask(OD3D_Task):
    """Base task that consumes an ObjectPairBatch and produces quant + qualit outputs."""

    def __init__(self, **kwargs):
        pass

    def forward(
        self,
        batch: ObjectPairBatch,
    ) -> Tuple[ObjectPairQuantBatch, ObjectPairQualitBatch]:
        raise NotImplementedError
