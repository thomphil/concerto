//! Tests for [`concerto_backend::process::build_command`].
//!
//! These tests never spawn any real processes — they build the `Command`
//! struct and inspect its program, arguments, and environment variables.

use std::ffi::OsStr;

use bytesize::ByteSize;
use concerto_backend::process::build_command;
use concerto_core::{EngineType, GpuId, ModelId, ModelSpec};
use tokio::process::Command;

fn spec_with_engine(engine: EngineType, args: Vec<&str>) -> ModelSpec {
    ModelSpec {
        id: ModelId::from("test-model"),
        name: "Test Model".to_string(),
        weight_path: "/models/test-model".to_string(),
        vram_required: ByteSize::gb(8),
        engine,
        engine_args: args.into_iter().map(String::from).collect(),
        pin: false,
    }
}

/// Extract the program name from a [`Command`] as a String.
fn program(cmd: &Command) -> String {
    cmd.as_std().get_program().to_string_lossy().into_owned()
}

/// Extract all arguments from a [`Command`] as owned Strings.
fn args(cmd: &Command) -> Vec<String> {
    cmd.as_std()
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect()
}

/// Look up an environment variable on a [`Command`].
fn env(cmd: &Command, key: &str) -> Option<String> {
    cmd.as_std()
        .get_envs()
        .find(|(k, _)| *k == OsStr::new(key))
        .and_then(|(_, v)| v.map(|v| v.to_string_lossy().into_owned()))
}

#[test]
fn vllm_command_shape() {
    let spec = spec_with_engine(
        EngineType::Vllm,
        vec!["--dtype", "float16", "--max-model-len", "4096"],
    );
    let cmd = build_command(&spec, GpuId(0), 8123);

    assert_eq!(program(&cmd), "python");
    let args = args(&cmd);
    assert!(args.contains(&"-m".to_string()));
    assert!(args.contains(&"vllm.entrypoints.openai.api_server".to_string()));
    assert!(args.contains(&"--model".to_string()));
    assert!(args.contains(&"/models/test-model".to_string()));
    assert!(args.contains(&"--port".to_string()));
    assert!(args.contains(&"8123".to_string()));
    // Engine args are appended verbatim.
    assert!(args.contains(&"--dtype".to_string()));
    assert!(args.contains(&"float16".to_string()));
    assert!(args.contains(&"--max-model-len".to_string()));
    assert!(args.contains(&"4096".to_string()));
}

#[test]
fn llama_cpp_command_uses_short_m_flag() {
    let spec = spec_with_engine(EngineType::LlamaCpp, vec!["--ctx-size", "4096"]);
    let cmd = build_command(&spec, GpuId(0), 8200);

    assert_eq!(program(&cmd), "llama-server");
    let args = args(&cmd);
    assert!(args.contains(&"-m".to_string()));
    assert!(args.contains(&"/models/test-model".to_string()));
    assert!(args.contains(&"--port".to_string()));
    assert!(args.contains(&"8200".to_string()));
    // llama.cpp should NOT use the long --model flag.
    assert!(!args.contains(&"--model".to_string()));
    assert!(args.contains(&"--ctx-size".to_string()));
}

#[test]
fn sglang_command_uses_model_path_flag() {
    let spec = spec_with_engine(EngineType::Sglang, vec![]);
    let cmd = build_command(&spec, GpuId(2), 8300);

    assert_eq!(program(&cmd), "python");
    let args = args(&cmd);
    assert!(args.contains(&"sglang.launch_server".to_string()));
    assert!(args.contains(&"--model-path".to_string()));
    assert!(args.contains(&"/models/test-model".to_string()));
    assert!(args.contains(&"--port".to_string()));
    assert!(args.contains(&"8300".to_string()));
}

#[test]
fn mock_engine_uses_mock_inference_backend_binary() {
    let spec = spec_with_engine(EngineType::Mock, vec!["--latency-ms", "5"]);
    let cmd = build_command(&spec, GpuId(0), 8400);

    assert_eq!(program(&cmd), "mock-inference-backend");
    let args = args(&cmd);
    assert!(args.contains(&"--port".to_string()));
    assert!(args.contains(&"8400".to_string()));
    assert!(args.contains(&"--latency-ms".to_string()));
    assert!(args.contains(&"5".to_string()));
}

#[test]
fn cuda_visible_devices_env_is_set_to_gpu_id() {
    let spec = spec_with_engine(EngineType::Vllm, vec![]);
    let cmd = build_command(&spec, GpuId(3), 8500);
    assert_eq!(env(&cmd, "CUDA_VISIBLE_DEVICES").as_deref(), Some("3"));

    let cmd_zero = build_command(&spec, GpuId(0), 8501);
    assert_eq!(env(&cmd_zero, "CUDA_VISIBLE_DEVICES").as_deref(), Some("0"));
}

#[test]
fn engine_args_preserve_order_after_builtin_args() {
    let spec = spec_with_engine(EngineType::Vllm, vec!["--first", "1", "--second", "2"]);
    let cmd = build_command(&spec, GpuId(0), 8600);
    let args = args(&cmd);

    let first = args.iter().position(|a| a == "--first").unwrap();
    let second = args.iter().position(|a| a == "--second").unwrap();
    let port = args.iter().position(|a| a == "--port").unwrap();

    assert!(port < first, "built-in args should come before engine args");
    assert!(first < second, "engine args should preserve order");
}
