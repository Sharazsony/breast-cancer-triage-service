"""Pydantic request/response models for the service contract."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

# The 30 dataset feature names with spaces replaced by underscores, in training order.
FEATURE_NAMES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension",
]

# Every feature is required, numeric (no strings), finite and >= 0.
NonNegative = Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]

PredictRequest = create_model(
    "PredictRequest",
    __config__=ConfigDict(extra="forbid"),
    **{name: (NonNegative, ...) for name in FEATURE_NAMES},
)


class PredictResponse(BaseModel):
    label: Literal["malignant", "benign"]
    malignant_probability: float
    threshold: float
    model_version: str
