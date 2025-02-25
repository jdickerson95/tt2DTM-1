"""Computational configuration for 2DTM."""

from typing import Annotated

import torch
from pydantic import BaseModel, Field, field_validator


class ComputationalConfig(BaseModel):
    """Serialization of computational resources allocated for 2DTM.

    Attributes
    ----------
    gpu_ids : list[int]
        Which GPU(s) to use for computation, defaults to 0 which will use device at
        index 0. A value of -2 or less corresponds to CPU device. A value of -1 will
        use all available GPUs.
    num_cpus : int
        Total number of CPUs to use, defaults to 1.
    """

    gpu_ids: int | list[int] = [0]
    num_cpus: Annotated[int, Field(ge=1)] = 1

    @field_validator("gpu_ids")  # type: ignore
    def validate_gpu_ids(cls, v):
        """Validate input value for GPU ids."""
        if isinstance(v, int):
            v = [v]

        # Check if -1 appears, it is only value in list
        if -1 in v and len(v) > 1:
            raise ValueError(
                "If -1 (all GPUs) is in the list, it must be the only value."
            )

        # Check if -2 appears, it is only value in list
        if -2 in v and len(v) > 1:
            raise ValueError("If -2 (CPU) is in the list, it must be the only value.")

        return v

    def gpu_devices(self) -> list[torch.device]:
        """Convert requested GPU IDs to torch device objects.

        Returns
        -------
        list[torch.device]
        """
        # Case where gpu_ids is integer
        if isinstance(self.gpu_ids, int):
            self.gpu_ids = [self.gpu_ids]

        if -1 in self.gpu_ids:
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        if -2 in self.gpu_ids:
            return [torch.device("cpu")]

        return [torch.device(f"cuda:{gpu_id}") for gpu_id in self.gpu_ids]
