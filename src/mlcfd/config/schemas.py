"""Pydantic configuration schemas for mesh, data IO, runs, and model parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, TypeAlias, get_args

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

ModelName: TypeAlias = Literal[
    "pca_svd",
    "pca_sklearn",
    "lle",
    "isomap",
    "kpca",
    "ae_spatial",
    "ae_temporal",
]


class MeshConfig(BaseModel):
    """Structured CFD mesh and cylinder geometry for mask construction."""

    nx: int = Field(gt=0, description="Number of grid nodes in x.")
    ny: int = Field(gt=0, description="Number of grid nodes in y.")
    lx: float = Field(gt=0, description="Domain length in x.")
    ly: float = Field(gt=0, description="Domain length in y.")
    x0: float = Field(description="Cylinder center x-coordinate.")
    y0: float = Field(description="Cylinder center y-coordinate.")
    r: float = Field(gt=0, description="Cylinder radius.")


class DataConfig(BaseModel):
    """Paths and options for loading snapshot matrices."""

    input_dir: Path = Field(description="Directory containing CSV snapshots.")
    reynolds_number: int = Field(gt=0, description="Reynolds number label used in filenames.")
    snapshot_filename_template: str = Field(
        default="modVcropRe{re}.csv",
        description="Filename template for snapshot CSV; must include `{re}` placeholder.",
    )
    test_size: float = Field(
        default=0.5, gt=0, lt=1, description="Fraction held out as test split."
    )
    spatial_reduction: bool = Field(
        default=True,
        description=(
            "If True, apply StandardScaler along the spatial snapshot axis used in notebooks."
        ),
    )
    random_seed: int = Field(
        default=42, description="Seed for any stochastic preprocessing or models."
    )

    def snapshot_csv_path(self) -> Path:
        """Return the resolved CSV path for the configured Reynolds number."""
        if "{re}" not in self.snapshot_filename_template:
            msg = "snapshot_filename_template must contain '{re}' placeholder"
            raise ValueError(msg)
        filename = self.snapshot_filename_template.format(re=self.reynolds_number)
        return self.input_dir / filename


class CropConfig(BaseModel):
    """Integer crop margins when building matrices from full binary fields."""

    top: int = Field(default=20, ge=0)
    bottom: int = Field(default=20, ge=0)
    left: int = Field(default=30, ge=0)
    right: int = Field(default=0, ge=0)
    time_start: int = Field(gt=0, description="First time index to read (inclusive).")
    time_end: int = Field(gt=0, description="Last time index bound (see reader semantics).")

    @field_validator("time_end")
    @classmethod
    def time_end_after_start(cls, value: int, info: ValidationInfo) -> int:
        """Ensure the configured time window is non-empty."""
        start = info.data.get("time_start")
        if isinstance(start, int) and value <= start:
            msg = "time_end must be greater than time_start"
            raise ValueError(msg)
        return value


class OutputConfig(BaseModel):
    """Where to write metrics, plots, and optional modal exports."""

    output_dir: Path = Field(description="Base directory for run artifacts.")
    save_modes: bool = Field(
        default=True, description="Whether to export spatial modes when supported."
    )
    save_plots: bool = Field(default=True, description="Whether to save diagnostic figures.")


class SweepModelConfig(BaseModel):
    """Parameters for models that sweep retained dimensions (PCA, etc.)."""

    profile: Literal["sweep"] = Field(
        default="sweep", description="Discriminator tag for sweep models."
    )
    r_max: int = Field(default=12, gt=0, description="Maximum number of components to evaluate.")
    r_step: int = Field(
        default=1, gt=0, description="Step size between component counts in sweeps."
    )


class ManifoldModelConfig(BaseModel):
    """Parameters for neighbor-graph manifold learners (LLE, Isomap)."""

    profile: Literal["manifold"] = Field(
        default="manifold",
        description="Discriminator tag for manifold models.",
    )
    r_max: int = Field(default=12, gt=0)
    r_step: int = Field(default=1, gt=0)
    k_neighbors: int = Field(default=10, gt=0)
    reg: float = Field(default=1e-9, gt=0)
    transpose_flag: bool = Field(
        default=False,
        description="If True, skip the notebook-style transpose before fitting.",
    )


class KPCAModelConfig(BaseModel):
    """Kernel PCA parameters including inverse transform settings."""

    profile: Literal["kpca"] = Field(
        default="kpca", description="Discriminator tag for KPCA models."
    )
    r_max: int = Field(default=12, gt=0)
    r_step: int = Field(default=1, gt=0)
    k_neighbors: int = Field(default=10, gt=0)
    reg: float = Field(default=1e-9, gt=0)
    transpose_flag: bool = Field(default=False)
    kernel: str = Field(default="rbf", description="Kernel name passed to KernelPCA.")
    gamma: float = Field(default=0.01)
    degree: int = Field(default=1, ge=0)
    alpha: float = Field(default=1e-3, gt=0)


class AutoencoderModelConfig(BaseModel):
    """Training hyperparameters for spatial or temporal autoencoders."""

    profile: Literal["autoencoder"] = Field(
        default="autoencoder",
        description="Discriminator tag for autoencoder models.",
    )
    batch_size: int = Field(default=128, gt=0)
    num_epochs: int = Field(default=500, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    scheduler_step: int = Field(default=3000, gt=0)
    scheduler_gamma: float = Field(default=0.1, gt=0)
    random_seed: int = Field(default=42)
    hidden_layers: list[int] = Field(
        default_factory=lambda: [8192, 2048, 512, 128, 32, 8],
        description="Encoder widths before the latent bottleneck (decoder mirrors).",
    )
    log_every_n_epochs: int = Field(
        default=25,
        ge=1,
        description="Log training progress every N epochs at INFO; DEBUG may log every epoch.",
    )


ModelParams: TypeAlias = Annotated[
    SweepModelConfig | ManifoldModelConfig | KPCAModelConfig | AutoencoderModelConfig,
    Field(discriminator="profile"),
]


class RunConfig(BaseModel):
    """Validated configuration for a full reduction run (CLI or notebooks)."""

    mesh: MeshConfig
    data: DataConfig
    crop: CropConfig | None = None
    output: OutputConfig
    model_name: ModelName
    model_params: ModelParams

    @model_validator(mode="after")
    def model_name_matches_params(self) -> RunConfig:
        """Ensure sweep vs manifold vs KPCA vs AE params align with the declared model id."""
        name = self.model_name
        params = self.model_params

        sweep_models = {"pca_svd", "pca_sklearn"}
        manifold_models = {"lle", "isomap"}
        kpca_models = {"kpca"}
        ae_models = {"ae_spatial", "ae_temporal"}

        if name in sweep_models and not isinstance(params, SweepModelConfig):
            msg = f"Model {name!r} requires SweepModelConfig parameters."
            raise ValueError(msg)
        if name in manifold_models and not isinstance(params, ManifoldModelConfig):
            msg = f"Model {name!r} requires ManifoldModelConfig parameters."
            raise ValueError(msg)
        if name in kpca_models and not isinstance(params, KPCAModelConfig):
            msg = f"Model {name!r} requires KPCAModelConfig parameters."
            raise ValueError(msg)
        if name in ae_models and not isinstance(params, AutoencoderModelConfig):
            msg = f"Model {name!r} requires AutoencoderModelConfig parameters."
            raise ValueError(msg)

        return self


def model_name_literal_values() -> tuple[str, ...]:
    """Return the allowed model identifiers as a runtime tuple."""
    return get_args(ModelName)
