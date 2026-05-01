//! Integration tests for parsing `concerto.toml` configurations.
//!
//! These tests exercise [`ConcertoConfig::from_toml_str`] with inline TOML
//! strings so they run without touching the filesystem.

use bytesize::ByteSize;
use concerto_config::{ConcertoConfig, ConfigError};
use concerto_core::{EngineType, EvictionPolicy, ModelId};

const FULL_CONFIG: &str = r#"
[server]
host = "127.0.0.1"
port = 9000

[routing]
eviction_policy = "lru"
cold_start_timeout_secs = 60
health_check_interval_secs = 5
max_healthy_temperature = 70
max_degraded_temperature = 80
vram_headroom = "2 GB"

[[models]]
id = "qwen2.5-7b"
name = "Qwen 2.5 7B"
weight_path = "/models/qwen2.5-7b"
vram_required = "14 GB"
engine = "vllm"
engine_args = ["--dtype", "float16", "--max-model-len", "4096"]

[[models]]
id = "phi-3-mini"
name = "Phi-3 Mini 3.8B"
weight_path = "/models/phi-3-mini"
vram_required = "8 GB"
engine = "llamacpp"
engine_args = ["--ctx-size", "4096"]

[[gpus]]
id = 0
max_temperature = 82

[[gpus]]
id = 1
"#;

#[test]
fn parses_full_happy_path_config() {
    let config = ConcertoConfig::from_toml_str(FULL_CONFIG).expect("config should parse");

    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.server.port, 9000);

    assert_eq!(config.routing.eviction_policy, EvictionPolicy::Lru);
    assert_eq!(config.routing.cold_start_timeout_secs, 60);
    assert_eq!(config.routing.health_check_interval_secs, 5);
    assert_eq!(config.routing.max_healthy_temperature, 70);
    assert_eq!(config.routing.max_degraded_temperature, 80);
    assert_eq!(config.routing.vram_headroom, ByteSize::gb(2));

    assert_eq!(config.models.len(), 2);
    assert_eq!(config.models[0].id, "qwen2.5-7b");
    assert_eq!(config.models[0].name, "Qwen 2.5 7B");
    assert_eq!(config.models[0].weight_path, "/models/qwen2.5-7b");
    assert_eq!(config.models[0].vram_required, ByteSize::gb(14));
    assert_eq!(config.models[0].engine, EngineType::Vllm);
    assert_eq!(
        config.models[0].engine_args,
        vec!["--dtype", "float16", "--max-model-len", "4096"]
    );
    assert_eq!(config.models[1].engine, EngineType::LlamaCpp);

    assert_eq!(config.gpus.len(), 2);
    assert_eq!(config.gpus[0].id, 0);
    assert_eq!(config.gpus[0].max_temperature, Some(82));
    assert_eq!(config.gpus[1].id, 1);
    assert_eq!(config.gpus[1].max_temperature, None);
}

#[test]
fn model_registry_contains_every_model() {
    let config = ConcertoConfig::from_toml_str(FULL_CONFIG).unwrap();
    let registry = config.model_registry();

    assert_eq!(registry.len(), 2);

    let qwen = registry
        .get(&ModelId("qwen2.5-7b".into()))
        .expect("qwen in registry");
    assert_eq!(qwen.name, "Qwen 2.5 7B");
    assert_eq!(qwen.weight_path, "/models/qwen2.5-7b");
    assert_eq!(qwen.vram_required, ByteSize::gb(14));
    assert_eq!(qwen.engine, EngineType::Vllm);

    assert!(registry.contains_key(&ModelId("phi-3-mini".into())));
}

#[test]
fn routing_config_maps_core_fields() {
    let config = ConcertoConfig::from_toml_str(FULL_CONFIG).unwrap();
    let core = config.routing_config();

    assert_eq!(core.eviction_policy, EvictionPolicy::Lru);
    assert_eq!(core.max_healthy_temperature, 70);
    assert_eq!(core.max_degraded_temperature, 80);
    assert_eq!(core.vram_headroom, ByteSize::gb(2));
}

#[test]
fn minimal_config_uses_all_defaults() {
    // The bare minimum: one model, one GPU, nothing else.
    let minimal = r#"
        [[models]]
        id = "tiny"
        name = "Tiny"
        weight_path = "/models/tiny"
        vram_required = "1 GB"
        engine = "mock"

        [[gpus]]
        id = 0
    "#;

    let config = ConcertoConfig::from_toml_str(minimal).expect("minimal config should parse");

    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 8000);

    assert_eq!(config.routing.eviction_policy, EvictionPolicy::Lru);
    assert_eq!(config.routing.cold_start_timeout_secs, 120);
    assert_eq!(config.routing.health_check_interval_secs, 10);
    assert_eq!(config.routing.max_healthy_temperature, 75);
    assert_eq!(config.routing.max_degraded_temperature, 85);
    assert_eq!(config.routing.vram_headroom, ByteSize::gb(1));

    assert_eq!(config.models[0].engine_args, Vec::<String>::new());
    assert_eq!(config.models[0].engine, EngineType::Mock);

    assert_eq!(config.gpus[0].max_temperature, None);
}

#[test]
fn duplicate_model_ids_are_rejected() {
    let toml = r#"
        [[models]]
        id = "dup"
        name = "First"
        weight_path = "/a"
        vram_required = "1 GB"
        engine = "mock"

        [[models]]
        id = "dup"
        name = "Second"
        weight_path = "/b"
        vram_required = "2 GB"
        engine = "mock"

        [[gpus]]
        id = 0
    "#;

    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    match err {
        ConfigError::DuplicateModelId(id) => assert_eq!(id, "dup"),
        other => panic!("expected DuplicateModelId, got {other:?}"),
    }
}

#[test]
fn duplicate_gpu_ids_are_rejected() {
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "mock"

        [[gpus]]
        id = 0

        [[gpus]]
        id = 0
    "#;

    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    match err {
        ConfigError::DuplicateGpuId(id) => assert_eq!(id, 0),
        other => panic!("expected DuplicateGpuId, got {other:?}"),
    }
}

#[test]
fn empty_models_section_is_rejected() {
    let toml = r#"
        [[gpus]]
        id = 0
    "#;

    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    assert!(matches!(err, ConfigError::EmptyModels));
}

#[test]
fn empty_gpus_section_is_rejected() {
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "mock"
    "#;

    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    assert!(matches!(err, ConfigError::EmptyGpus));
}

#[test]
fn invalid_engine_type_produces_parse_error() {
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "not-a-real-engine"

        [[gpus]]
        id = 0
    "#;

    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    assert!(
        matches!(err, ConfigError::Parse(_)),
        "expected Parse error, got {err:?}"
    );
}

#[test]
fn port_zero_is_rejected() {
    let toml = r#"
        [server]
        port = 0

        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "mock"

        [[gpus]]
        id = 0
    "#;

    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    assert!(matches!(err, ConfigError::InvalidPort(0)));
}

#[test]
fn per_gpu_max_temperature_override_is_preserved() {
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "mock"

        [[gpus]]
        id = 0
        max_temperature = 82

        [[gpus]]
        id = 1
        max_temperature = 90

        [[gpus]]
        id = 2
    "#;

    let config = ConcertoConfig::from_toml_str(toml).unwrap();
    assert_eq!(config.gpus[0].max_temperature, Some(82));
    assert_eq!(config.gpus[1].max_temperature, Some(90));
    assert_eq!(config.gpus[2].max_temperature, None);
}

#[test]
fn bytesize_string_parses_to_expected_byte_count() {
    // Whichever base bytesize v2 uses for "GB", it must be consistent with
    // the `ByteSize::gb(..)` constructor — i.e. round-tripping through the
    // same crate. This test pins that consistency.
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "14 GB"
        engine = "mock"

        [[gpus]]
        id = 0
    "#;

    let config = ConcertoConfig::from_toml_str(toml).unwrap();
    assert_eq!(config.models[0].vram_required, ByteSize::gb(14));
    assert_eq!(
        config.models[0].vram_required.as_u64(),
        ByteSize::gb(14).as_u64()
    );
}

#[test]
fn missing_required_model_field_produces_parse_error() {
    // `vram_required` is required and has no default.
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        engine = "mock"

        [[gpus]]
        id = 0
    "#;

    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    assert!(
        matches!(err, ConfigError::Parse(_)),
        "expected Parse error, got {err:?}"
    );
}

#[test]
fn malformed_toml_produces_parse_error() {
    let toml = "this is ::: not valid toml == ==";
    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    assert!(matches!(err, ConfigError::Parse(_)));
}

#[test]
fn parses_custom_engine_variant() {
    let toml = r#"
        [[models]]
        id = "my-custom"
        name = "Custom Server"
        weight_path = "/models/custom"
        vram_required = "8 GB"
        engine = { custom = { command = "my-inference-server", args = ["--weights", "/models/custom", "--port", "{port}"], health_endpoint = "/ready" } }

        [[gpus]]
        id = 0
    "#;

    let config = ConcertoConfig::from_toml_str(toml).expect("config should parse");
    assert_eq!(config.models.len(), 1);
    match &config.models[0].engine {
        EngineType::Custom {
            command,
            args,
            health_endpoint,
        } => {
            assert_eq!(command, "my-inference-server");
            assert_eq!(args.len(), 4);
            assert_eq!(args[3], "{port}");
            assert_eq!(health_endpoint, "/ready");
        }
        other => panic!("expected EngineType::Custom, got {:?}", other),
    }
}

#[test]
fn valid_max_vram_fraction_parses_and_propagates_to_model_spec() {
    let toml = r#"
        [[models]]
        id = "qwen-shared"
        name = "Qwen shared"
        weight_path = "/models/qwen"
        vram_required = "2 GB"
        engine = "vllm"
        max_vram_fraction = 0.5

        [[gpus]]
        id = 0
    "#;

    let config = ConcertoConfig::from_toml_str(toml).expect("config should parse");
    assert_eq!(config.models[0].max_vram_fraction, Some(0.5));

    let registry = config.model_registry();
    let spec = registry.get(&ModelId("qwen-shared".into())).unwrap();
    assert_eq!(spec.max_vram_fraction, Some(0.5));
}

#[test]
fn missing_max_vram_fraction_defaults_to_none() {
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "vllm"

        [[gpus]]
        id = 0
    "#;

    let config = ConcertoConfig::from_toml_str(toml).unwrap();
    assert_eq!(config.models[0].max_vram_fraction, None);
}

#[test]
fn out_of_range_max_vram_fraction_is_rejected() {
    for bad in [-0.1_f64, 1.1_f64, 2.0_f64] {
        let toml = format!(
            r#"
            [[models]]
            id = "m"
            name = "M"
            weight_path = "/m"
            vram_required = "1 GB"
            engine = "vllm"
            max_vram_fraction = {bad}

            [[gpus]]
            id = 0
            "#
        );
        let err =
            ConcertoConfig::from_toml_str(&toml).expect_err(&format!("{bad} must be rejected"));
        match err {
            ConfigError::InvalidVramFraction { model, value } => {
                assert_eq!(model, "m");
                assert!((value - bad).abs() < f64::EPSILON);
            }
            other => panic!("expected InvalidVramFraction for {bad}, got {other:?}"),
        }
    }
}

#[test]
fn nan_max_vram_fraction_is_rejected() {
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "vllm"
        max_vram_fraction = nan

        [[gpus]]
        id = 0
    "#;
    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    match err {
        ConfigError::InvalidVramFraction { model, value } => {
            assert_eq!(model, "m");
            assert!(value.is_nan());
        }
        other => panic!("expected InvalidVramFraction, got {other:?}"),
    }
}

#[test]
fn zero_max_vram_fraction_is_rejected() {
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "vllm"
        max_vram_fraction = 0.0

        [[gpus]]
        id = 0
    "#;
    let err = ConcertoConfig::from_toml_str(toml).unwrap_err();
    match err {
        ConfigError::InvalidVramFraction { model, value } => {
            assert_eq!(model, "m");
            assert_eq!(value, 0.0);
        }
        other => panic!("expected InvalidVramFraction, got {other:?}"),
    }
}

#[test]
fn upper_bound_max_vram_fraction_is_accepted() {
    // Closed upper bound: 1.0 means "the whole GPU" and is legitimate.
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "vllm"
        max_vram_fraction = 1.0

        [[gpus]]
        id = 0
    "#;
    let config = ConcertoConfig::from_toml_str(toml).expect("1.0 must parse");
    assert_eq!(config.models[0].max_vram_fraction, Some(1.0));
}

#[test]
fn max_vram_fraction_with_explicit_engine_arg_still_validates() {
    // When both are set, validation accepts the config (it's a config-load
    // warning, not an error). The build_command tests verify that the
    // explicit `engine_args` value wins over `max_vram_fraction`.
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "vllm"
        engine_args = ["--gpu-memory-utilization", "0.7"]
        max_vram_fraction = 0.5

        [[gpus]]
        id = 0
    "#;
    let config = ConcertoConfig::from_toml_str(toml).expect("both-set is a warning, not an error");
    assert_eq!(config.models[0].max_vram_fraction, Some(0.5));
    assert_eq!(
        config.models[0].engine_args,
        vec!["--gpu-memory-utilization", "0.7"]
    );
}

#[test]
fn max_vram_fraction_on_non_vllm_engine_validates_with_warning() {
    // Setting the field on a non-vLLM engine logs a warning but does not
    // reject the config — operators may set it consistently across engines
    // in mixed deployments.
    let toml = r#"
        [[models]]
        id = "m"
        name = "M"
        weight_path = "/m"
        vram_required = "1 GB"
        engine = "llamacpp"
        max_vram_fraction = 0.5

        [[gpus]]
        id = 0
    "#;
    let config = ConcertoConfig::from_toml_str(toml).expect("non-vLLM engines must not error");
    assert_eq!(config.models[0].max_vram_fraction, Some(0.5));
    assert_eq!(config.models[0].engine, EngineType::LlamaCpp);
}

#[test]
fn existing_string_engines_still_parse_after_custom_variant_added() {
    // Regression test — the addition of the Custom struct variant must not
    // break the bare-string form used for built-in engines.
    for (engine_str, expected) in [
        ("vllm", EngineType::Vllm),
        ("llamacpp", EngineType::LlamaCpp),
        ("sglang", EngineType::Sglang),
        ("mock", EngineType::Mock),
    ] {
        let toml = format!(
            r#"
            [[models]]
            id = "m"
            name = "M"
            weight_path = "/m"
            vram_required = "1 GB"
            engine = "{engine_str}"

            [[gpus]]
            id = 0
            "#
        );
        let config =
            ConcertoConfig::from_toml_str(&toml).expect("{engine_str} bare form should parse");
        assert_eq!(config.models[0].engine, expected);
    }
}
