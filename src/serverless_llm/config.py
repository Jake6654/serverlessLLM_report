"""
파일 읽기
→ YAML parsing
→ Dataclass 생성
→ validate_config()
→ 유효하면 반환
→ 잘못됐으면 ConfigValidationError
"""

"""Typed configuration models for the experiment."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentSettings:
    """Settings that apply to the complete experiment."""

    name: str
    random_seed: int
    repetitions: int


@dataclass(frozen=True)
class ModelConfig:
    """LLM and vLLM model configuration."""

    name: str
    revision: str
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    kv_cache_dtype: str


@dataclass(frozen=True)
class ServerConfig:
    """Settings for starting and connecting to the vLLM server."""

    host: str
    port: int
    startup_timeout_seconds: int
    request_timeout_seconds: int
    readiness_poll_interval_seconds: float


@dataclass(frozen=True)
class RequestConfig:
    """Payload shared by every policy and workload request."""

    endpoint: str
    prompt: str
    max_tokens: int
    temperature: float
    stream: bool


@dataclass(frozen=True)
class MonitoringConfig:
    """GPU monitoring settings."""

    gpu_sample_interval_ms: int


@dataclass(frozen=True)
class WorkloadConfig:
    """Initial synthetic workload settings."""

    type: str
    request_interval_seconds: float
    total_requests: int
    concurrency: int


@dataclass(frozen=True)
class FixedKeepWarmConfig:
    """Configuration for the fixed keep-warm policy."""

    timeout_seconds: int
    timeout_candidates_seconds: tuple[int, ...]


@dataclass(frozen=True)
class RuleBasedConfig:
    """Configuration for the EMA-based adaptive policy."""

    alpha: float
    timeout_multiplier: float
    minimum_timeout_seconds: int
    maximum_timeout_seconds: int


@dataclass(frozen=True)
class MLBasedConfig:
    """Configuration for the ML-based adaptive policy."""

    model_type: str
    prediction_horizon_seconds: int
    decision_threshold: float


@dataclass(frozen=True)
class PoliciesConfig:
    """Configuration for policies that require tunable parameters."""

    fixed_keep_warm: FixedKeepWarmConfig
    rule_based: RuleBasedConfig
    ml_based: MLBasedConfig


@dataclass(frozen=True)
class MetricsConfig:
    """Thresholds used to calculate experiment metrics."""

    ttft_slo_seconds: float
    gpu_active_memory_threshold_mib: int


@dataclass(frozen=True)
class OutputConfig:
    """Locations for generated experiment artifacts."""

    results_directory: Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete typed configuration for one experiment run."""

    experiment: ExperimentSettings
    model: ModelConfig
    server: ServerConfig
    request: RequestConfig
    monitoring: MonitoringConfig
    workload: WorkloadConfig
    policies: PoliciesConfig
    metrics: MetricsConfig
    output: OutputConfig


class ConfigLoadError(ValueError):
    """Raised when an experiment configuration cannot be loaded."""


class ConfigValidationError(ValueError):
    """Raised when configuration values violate experiment rules."""


def _is_integer(value: object) -> bool:
    """Return True only for integers, excluding booleans."""

    return type(value) is int


def _is_number(value: object) -> bool:
    """Return True only for integer or floating-point numbers."""

    return type(value) in (int, float)


def _is_nonempty_string(value: object) -> bool:
    """Return True for strings containing non-whitespace characters."""

    return isinstance(value, str) and bool(value.strip())


def validate_config(config: ExperimentConfig) -> None:
    """Validate types, ranges, and relationships between settings."""

    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    experiment = config.experiment
    model = config.model
    server = config.server
    request = config.request
    monitoring = config.monitoring
    workload = config.workload
    fixed = config.policies.fixed_keep_warm
    rule = config.policies.rule_based
    ml = config.policies.ml_based
    metrics = config.metrics
    output = config.output

    # Experiment settings
    check(
        _is_nonempty_string(experiment.name),
        "experiment.name must be a non-empty string",
    )
    check(
        _is_integer(experiment.random_seed),
        "experiment.random_seed must be an integer",
    )
    check(
        _is_integer(experiment.repetitions)
        and experiment.repetitions >= 1,
        "experiment.repetitions must be an integer of at least 1",
    )

    # Model settings
    check(
        _is_nonempty_string(model.name),
        "model.name must be a non-empty string",
    )
    check(
        _is_nonempty_string(model.revision),
        "model.revision must be a non-empty string",
    )
    check(
        isinstance(model.dtype, str)
        and model.dtype in {"bfloat16", "float16", "float32"},
        "model.dtype must be bfloat16, float16, or float32",
    )
    check(
        _is_integer(model.max_model_len) and model.max_model_len > 0,
        "model.max_model_len must be a positive integer",
    )
    check(
        _is_number(model.gpu_memory_utilization)
        and 0 < model.gpu_memory_utilization <= 1,
        "model.gpu_memory_utilization must be greater than 0 and at most 1",
    )
    check(
        _is_integer(model.max_num_seqs) and model.max_num_seqs > 0,
        "model.max_num_seqs must be a positive integer",
    )
    check(
        _is_nonempty_string(model.kv_cache_dtype),
        "model.kv_cache_dtype must be a non-empty string",
    )

    # Server settings
    check(
        _is_nonempty_string(server.host),
        "server.host must be a non-empty string",
    )
    check(
        _is_integer(server.port) and 1 <= server.port <= 65535,
        "server.port must be an integer between 1 and 65535",
    )
    check(
        _is_number(server.startup_timeout_seconds)
        and server.startup_timeout_seconds > 0,
        "server.startup_timeout_seconds must be positive",
    )
    check(
        _is_number(server.request_timeout_seconds)
        and server.request_timeout_seconds > 0,
        "server.request_timeout_seconds must be positive",
    )
    check(
        _is_number(server.readiness_poll_interval_seconds)
        and server.readiness_poll_interval_seconds > 0,
        "server.readiness_poll_interval_seconds must be positive",
    )

    # Request settings
    check(
        _is_nonempty_string(request.endpoint)
        and request.endpoint.startswith("/"),
        "request.endpoint must be a non-empty path starting with '/'",
    )
    check(
        _is_nonempty_string(request.prompt),
        "request.prompt must be a non-empty string",
    )
    check(
        _is_integer(request.max_tokens) and request.max_tokens > 0,
        "request.max_tokens must be a positive integer",
    )
    check(
        _is_number(request.temperature) and request.temperature >= 0,
        "request.temperature must be a non-negative number",
    )
    check(type(request.stream) is bool, "request.stream must be a boolean")

    # Monitoring settings
    check(
        _is_integer(monitoring.gpu_sample_interval_ms)
        and monitoring.gpu_sample_interval_ms > 0,
        "monitoring.gpu_sample_interval_ms must be a positive integer",
    )

    # Workload settings
    check(
        isinstance(workload.type, str)
        and workload.type in {"steady", "bursty", "sparse", "mixed"},
        "workload.type must be steady, bursty, sparse, or mixed",
    )
    check(
        _is_number(workload.request_interval_seconds)
        and workload.request_interval_seconds > 0,
        "workload.request_interval_seconds must be positive",
    )
    check(
        _is_integer(workload.total_requests) and workload.total_requests > 0,
        "workload.total_requests must be a positive integer",
    )
    check(
        _is_integer(workload.concurrency) and workload.concurrency > 0,
        "workload.concurrency must be a positive integer",
    )
    if _is_integer(workload.concurrency) and _is_integer(
        workload.total_requests
    ):
        check(
            workload.concurrency <= workload.total_requests,
            "workload.concurrency cannot exceed total_requests",
        )

    # Fixed keep-warm settings
    check(
        _is_integer(fixed.timeout_seconds) and fixed.timeout_seconds > 0,
        "policies.fixed_keep_warm.timeout_seconds must be positive",
    )
    candidates_are_tuple = isinstance(
        fixed.timeout_candidates_seconds, tuple
    )
    check(
        candidates_are_tuple and bool(fixed.timeout_candidates_seconds),
        "fixed keep-warm timeout candidates must be a non-empty tuple",
    )
    if candidates_are_tuple:
        check(
            all(
                _is_integer(value) and value > 0
                for value in fixed.timeout_candidates_seconds
            ),
            "all fixed keep-warm timeout candidates must be positive integers",
        )
        check(
            fixed.timeout_seconds in fixed.timeout_candidates_seconds,
            "fixed keep-warm timeout must appear in timeout candidates",
        )

    # Rule-based settings
    check(
        _is_number(rule.alpha) and 0 < rule.alpha <= 1,
        "policies.rule_based.alpha must be greater than 0 and at most 1",
    )
    check(
        _is_number(rule.timeout_multiplier)
        and rule.timeout_multiplier > 0,
        "policies.rule_based.timeout_multiplier must be positive",
    )
    check(
        _is_number(rule.minimum_timeout_seconds)
        and rule.minimum_timeout_seconds > 0,
        "rule-based minimum timeout must be positive",
    )
    check(
        _is_number(rule.maximum_timeout_seconds)
        and rule.maximum_timeout_seconds > 0,
        "rule-based maximum timeout must be positive",
    )
    if _is_number(rule.minimum_timeout_seconds) and _is_number(
        rule.maximum_timeout_seconds
    ):
        check(
            rule.maximum_timeout_seconds >= rule.minimum_timeout_seconds,
            "rule-based maximum timeout cannot be smaller than minimum timeout",
        )

    # ML-based settings
    check(
        isinstance(ml.model_type, str)
        and ml.model_type
        in {"logistic_regression", "random_forest", "xgboost"},
        "ML model type must be logistic_regression, random_forest, or xgboost",
    )
    check(
        _is_integer(ml.prediction_horizon_seconds)
        and ml.prediction_horizon_seconds > 0,
        "ML prediction horizon must be a positive integer",
    )
    check(
        _is_number(ml.decision_threshold)
        and 0 <= ml.decision_threshold <= 1,
        "ML decision threshold must be between 0 and 1",
    )

    # Metric settings
    check(
        _is_number(metrics.ttft_slo_seconds)
        and metrics.ttft_slo_seconds > 0,
        "metrics.ttft_slo_seconds must be positive",
    )
    check(
        _is_integer(metrics.gpu_active_memory_threshold_mib)
        and metrics.gpu_active_memory_threshold_mib >= 0,
        "GPU active memory threshold must be a non-negative integer",
    )

    # Output settings
    check(
        isinstance(output.results_directory, Path)
        and output.results_directory != Path("."),
        "output.results_directory must be a non-empty path",
    )
    check(
        isinstance(output.results_directory, Path)
        and not output.results_directory.is_absolute(),
        "output.results_directory must be relative to the project",
    )

    if errors:
        formatted_errors = "\n".join(
            f"- {message}" for message in errors
        )
        raise ConfigValidationError(
            f"Invalid experiment configuration:\n{formatted_errors}"
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A complete immutable and validated experiment configuration.

    Raises:
        ConfigLoadError: If the file cannot be read, the YAML is malformed,
            or the expected configuration structure is missing.
        ConfigValidationError: If values violate experiment rules.
    """

    config_path = Path(path)

    try:
        yaml_text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigLoadError(
            f"Could not read configuration file {config_path}: {error}"
        ) from error

    try:
        raw_config = yaml.safe_load(yaml_text)
    except yaml.YAMLError as error:
        raise ConfigLoadError(
            f"Invalid YAML in {config_path}: {error}"
        ) from error

    if not isinstance(raw_config, dict):
        raise ConfigLoadError(
            f"Configuration root in {config_path} must be a mapping"
        )

    try:
        policies_data = raw_config["policies"]
        fixed_data = policies_data["fixed_keep_warm"]

        fixed_keep_warm = FixedKeepWarmConfig(
            timeout_seconds=fixed_data["timeout_seconds"],
            timeout_candidates_seconds=tuple(
                fixed_data["timeout_candidates_seconds"]
            ),
        )
        policies = PoliciesConfig(
            fixed_keep_warm=fixed_keep_warm,
            rule_based=RuleBasedConfig(**policies_data["rule_based"]),
            ml_based=MLBasedConfig(**policies_data["ml_based"]),
        )
        output = OutputConfig(
            results_directory=Path(
                raw_config["output"]["results_directory"]
            )
        )
        config = ExperimentConfig(
            experiment=ExperimentSettings(**raw_config["experiment"]),
            model=ModelConfig(**raw_config["model"]),
            server=ServerConfig(**raw_config["server"]),
            request=RequestConfig(**raw_config["request"]),
            monitoring=MonitoringConfig(**raw_config["monitoring"]),
            workload=WorkloadConfig(**raw_config["workload"]),
            policies=policies,
            metrics=MetricsConfig(**raw_config["metrics"]),
            output=output,
        )
    except (KeyError, TypeError) as error:
        raise ConfigLoadError(
            f"Invalid configuration structure in {config_path}: {error}"
        ) from error

    validate_config(config)
    return config
