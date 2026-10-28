import os
import textwrap

from config import Config


def _write_cfg(tmp_path, body: str):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _fresh(tmp_path, body):
    # Config 是单例，测试间要重置内部状态
    cfg = Config()
    cfg._config = {}
    cfg.load(str(_write_cfg(tmp_path, body)))
    return cfg


def test_override_creates_nested_path(tmp_path):
    cfg = _fresh(tmp_path, "llm:\n  model_name: A\n")
    cfg.override("narration.explain.mode", "remix")
    assert cfg.narration["explain"]["mode"] == "remix"


def test_deep_update_merges_without_clobbering_siblings(tmp_path):
    cfg = _fresh(tmp_path, """
        narration:
          explain:
            mode: picture_led
            target_minutes: 7
        commentary:
          font_size: 40
    """)
    cfg.deep_update({"narration": {"explain": {"target_minutes": 5}},
                     "commentary": {"font_size": 44}})
    ex = cfg.narration["explain"]
    assert ex["target_minutes"] == 5
    assert ex["mode"] == "picture_led"  # 兄弟键保留
    assert cfg.commentary["font_size"] == 44


def test_env_var_substitution(tmp_path):
    os.environ["VE_TEST_KEY"] = "secret123"
    try:
        cfg = _fresh(tmp_path, "llm:\n  api_key: ${VE_TEST_KEY}\n")
        assert cfg.llm["api_key"] == "secret123"
    finally:
        del os.environ["VE_TEST_KEY"]
