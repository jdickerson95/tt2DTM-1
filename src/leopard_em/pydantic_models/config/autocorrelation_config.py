"""Serialization and validation of autocorrelation search parameters for 2DTM."""

from leopard_em.pydantic_models.custom_types import BaseModel2DTM


class AutocorrelationConfig(BaseModel2DTM):
    """Serialization and validation of autocorrelation search parameters for 2DTM.

    Attributes
    ----------
    enabled : bool
        Whether to enable autocorrelation search. Default is False.
    autocorrelation_map : str
        Path to the autocorrelation 3D map file.

    """

    enabled: bool = False
    autocorrelation_map: str = ""
