import os
import re
import logging
import yaml
from pathlib import Path


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config = {}
        self._logger = None

    def load(self, config_path: str = "config.yaml"):
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        self._config = self._substitute_env_vars(raw_config)
        self._setup_logging()
        self._resolve_external_text(config_file.parent)
        self._log_loaded_config()

    def _resolve_external_text(self, base_dir: Path):
        """支持把超长文本放到外部文件：video.series_background_file 指定路径时，
        读取其内容作为 series_background（优先级高于内联值）。路径先按相对 config
        目录解析，再回退到当前工作目录。找不到则保留内联值并告警。"""
        video = self._config.get("video", {})
        bg_file = video.get("series_background_file")
        if not bg_file:
            return
        path = Path(bg_file)
        if not path.is_absolute():
            candidate = base_dir / bg_file
            path = candidate if candidate.exists() else Path(bg_file)
        if path.exists():
            video["series_background"] = path.read_text(encoding="utf-8")
            self._logger.info(
                f"Loaded series_background from file: {path} "
                f"({len(video['series_background'])} chars)")
        else:
            self._logger.warning(
                f"series_background_file not found: {bg_file} (falling back to inline value)")

    def _substitute_env_vars(self, obj):
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, obj)
            result = obj
            for match in matches:
                env_value = os.environ.get(match, "")
                result = result.replace(f"${{{match}}}", env_value)
            return result
        return obj

    def _setup_logging(self):
        log_config = self._config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper())
        fmt = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        logging.basicConfig(level=level, format=fmt, force=True)
        self._logger = logging.getLogger("VideoEditor")

    def _log_loaded_config(self):
        self._logger.info("Configuration loaded successfully")
        self._logger.info(f"Input directory: {self._config.get('video', {}).get('input_dir')}")
        self._logger.info(f"Max parts per video: {self._config.get('max_parts', 1)}")
        self._logger.info(f"Output formats: {len(self._config.get('output_formats', []))}")
        for i, fmt in enumerate(self._config.get("output_formats", [])):
            self._logger.info(f"  Format {i+1}: {fmt.get('aspect_ratio')} {fmt.get('target_resolution')} -> {fmt.get('suffix')}")
        self._logger.info(f"Transition: {self._config.get('transition', {}).get('duration', 0.5)}s {self._config.get('transition', {}).get('type', 'crossfade')}")
        self._logger.info(f"ASR model: {self._config.get('asr', {}).get('model')}")
        self._logger.info(f"LLM base_url: {self._config.get('llm', {}).get('base_url')}")
        self._logger.info(f"LLM model: {self._config.get('llm', {}).get('model_name')}")
        self._logger.info(f"Target duration: {self._config.get('llm', {}).get('target_duration')} minutes")
        strategy = self._config.get("strategy", {})
        if strategy:
            self._logger.info(f"Strategy: target={strategy.get('target_duration')}min, range={strategy.get('min_duration')}-{strategy.get('max_duration')}min, clip={strategy.get('min_clip_duration')}-{strategy.get('max_clip_duration')}s, max_clips={strategy.get('max_clips')}")
        ending = self._config.get("ending", {})
        if ending.get("enabled", True):
            self._logger.info(f"Ending: enabled, fade={ending.get('fade_duration')}s, text='{ending.get('text')}', display={ending.get('display_duration')}s")
        self._logger.info(f"Subtitle enabled: {self._config.get('subtitle', {}).get('enabled')}")
        self._logger.info(f"Subtitle font: {self._config.get('subtitle', {}).get('font_path')}")
        author = self._config.get("author", {})
        self._logger.info(f"Author: {author.get('enabled') and author.get('name') or 'disabled'}")
        desc = self._config.get("description", {})
        if desc.get("enabled", True) and desc.get("font_size"):
            self._logger.info(f"Description: enabled, size={desc.get('font_size')}, color={desc.get('font_color')}, y_offset={desc.get('y_offset')}")
        nar = self._config.get("narration", {})
        if nar.get("enabled", False):
            tts = nar.get("tts", {})
            ex = nar.get("explain", {})
            engine = str(tts.get("engine", "volcano")).strip().lower()
            if engine == "edge":
                voice_desc = tts.get("edge", {}).get("voice", "")
            else:
                voice_desc = tts.get("volcano", {}).get("speaker", "")
            self._logger.info(
                f"Narration(explain): TTS={engine} voice={voice_desc}, "
                f"target={ex.get('target_minutes', 7)}min, series={nar.get('continuity', {}).get('series_name', '')}"
            )

    @property
    def video(self):
        return self._config.get("video", {})

    @property
    def asr(self):
        return self._config.get("asr", {})

    @property
    def llm(self):
        return self._config.get("llm", {})

    @property
    def subtitle(self):
        return self._config.get("subtitle", {})

    @property
    def author(self):
        return self._config.get("author", {})

    @property
    def narration(self):
        return self._config.get("narration", {})

    @property
    def naming(self):
        return self._config.get("naming", {})

    @property
    def description(self):
        return self._config.get("description", {})

    @property
    def transition(self):
        return self._config.get("transition", {})

    @property
    def output_formats(self):
        return self._config.get("output_formats", [])

    @property
    def max_parts(self):
        return self._config.get("max_parts", 1)

    @property
    def strategy(self):
        return self._config.get("strategy", {})

    @property
    def ending(self):
        return self._config.get("ending", {})

    @property
    def logging(self):
        return self._config.get("logging", {})

    @property
    def bgm(self):
        return self._config.get("bgm", {})

    @property
    def copyright_notice(self):
        return self._config.get("copyright_notice", {})

    @property
    def commentary(self):
        return self._config.get("commentary", {})

    @property
    def scene_notes(self):
        return self._config.get("scene_notes", {})

    @property
    def faces(self):
        return self._config.get("faces", {})

    # ── 运行期覆盖（webui/CLI 用，不回写 yaml）────────────────────────

    def override(self, dotted_path: str, value):
        """按点分路径覆盖单个配置项，如 override("narration.explain.mode", "remix")。"""
        keys = dotted_path.split(".")
        node = self._config
        for k in keys[:-1]:
            nxt = node.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                node[k] = nxt
            node = nxt
        node[keys[-1]] = value

    def deep_update(self, overrides: dict):
        """递归合并 overrides 到当前配置（dict 深合并，其他类型直接覆盖）。"""

        def _merge(dst: dict, src: dict):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _merge(dst[k], v)
                else:
                    dst[k] = v

        if isinstance(overrides, dict):
            _merge(self._config, overrides)

    def get_logger(self, name: str = None):
        if name:
            return logging.getLogger(f"VideoEditor.{name}")
        return self._logger


_config = Config()


def load_config(config_path: str = "config.yaml") -> Config:
    _config.load(config_path)
    return _config

