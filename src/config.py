"""Settings loaded from ``config/settings.yaml`` (pydantic-settings), overridable from the env.

Every knob and seed in the project lives in that YAML file. Environment variables override it
with the prefix ``VS101_`` and ``__`` as the nesting delimiter, e.g. ``VS101_EVAL__K=20``.
Defaults below mirror the YAML so the models also work stand-alone (tests build them in code).
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


class RealDatasetConfig(BaseModel):
    name: str = "news"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_n: int = 50_000
    batch_size: int = 128
    min_chars: int = 40
    max_chars: int = 300


class SyntheticDatasetConfig(BaseModel):
    name: str
    n: int
    d: int
    n_clusters: int = 1
    intrinsic_dim: int
    cluster_std: float = 1.0
    noise: float = 0.0


class DatasetsConfig(BaseModel):
    n_queries: int = 1000
    k_gt: int = 100
    real: RealDatasetConfig = Field(default_factory=RealDatasetConfig)
    synthetic: list[SyntheticDatasetConfig] = Field(default_factory=list)


class EvalConfig(BaseModel):
    k: int = 10


class ExactConfig(BaseModel):
    block_mb: float = 256.0


class ScalingConfig(BaseModel):
    sizes: list[int] = Field(default_factory=lambda: [10_000, 30_000, 100_000])
    d: int = 384
    n_queries: int = 50
    extrapolate_to: list[int] = Field(default_factory=lambda: [1_000_000])


class IVFConfig(BaseModel):
    nlist: int = 256
    nprobe_sweep: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256])
    train_points_per_centroid: int = 64
    kmeans_max_iter: int = 20
    toy_nlist: int = 12


class PQConfig(BaseModel):
    n_subvectors: int = 48
    n_subvectors_sweep: list[int] = Field(default_factory=lambda: [8, 16, 32, 48, 96])
    nbits: int = 8
    train_size: int = 20_000
    kmeans_max_iter: int = 20


class IVFPQConfig(BaseModel):
    nlist: int = 256
    n_subvectors: int = 48
    nprobe_sweep: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256])


class HNSWConfig(BaseModel):
    m: int = 16
    m_sweep: list[int] = Field(default_factory=lambda: [4, 8, 16, 32])
    ef_construction: int = 64
    ef_search: int = 64
    ef_search_sweep: list[int] = Field(default_factory=lambda: [10, 16, 32, 64, 128, 256])
    max_scratch_n: int = 20_000
    sweep_n: int = 10_000
    ladder: list[int] = Field(default_factory=lambda: [1000, 2000, 4000, 8000, 16000])


class LSHConfig(BaseModel):
    rows: int = 10
    bands: int = 16
    bands_sweep: list[int] = Field(default_factory=lambda: [2, 4, 8, 16, 32, 64])


class TriangleConfig(BaseModel):
    scratch_n: int = 20_000
    nlist: int = 128
    hnswlib_m_sweep: list[int] = Field(default_factory=lambda: [8, 16, 32])


class DecisionConfig(BaseModel):
    sizes: list[int] = Field(default_factory=lambda: [12_500, 25_000, 50_000])
    target_recall: float = 0.95
    ef_candidates: list[int] = Field(default_factory=lambda: [16, 32, 64, 128, 256])
    nprobe_candidates: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    sample_sizes: list[int] = Field(default_factory=lambda: [25, 50, 100, 200, 400])
    sample_repeats: int = 30


class BenchConfig(BaseModel):
    threads: int = 1
    warmup: int = 3
    max_queries: int = 300


class Settings(BaseSettings):
    """The whole project configuration. Use :func:`get_settings` rather than building it."""

    model_config = SettingsConfigDict(
        yaml_file=CONFIG_PATH,
        env_prefix="VS101_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    seed: int = 42
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    exact: ExactConfig = Field(default_factory=ExactConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    ivf: IVFConfig = Field(default_factory=IVFConfig)
    pq: PQConfig = Field(default_factory=PQConfig)
    ivfpq: IVFPQConfig = Field(default_factory=IVFPQConfig)
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)
    lsh: LSHConfig = Field(default_factory=LSHConfig)
    triangle: TriangleConfig = Field(default_factory=TriangleConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit kwargs > environment > YAML file > field defaults.
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The cached project settings (YAML + environment overrides)."""
    return Settings()
