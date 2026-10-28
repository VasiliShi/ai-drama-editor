import os
import gc
import json
from pathlib import Path
from typing import Tuple


def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _patch_torch_load():
    """torch 2.6+ 把 torch.load 的 weights_only 默认改成 True，会导致
    whisperx VAD / pyannote 的官方 checkpoint(含 omegaconf 对象)加载失败。
    这些权重来自可信官方仓库，全局放开 weights_only=False。幂等。"""
    import torch
    if getattr(torch.load, "_wo_patched", False):
        return
    _orig = torch.load

    def _patched(*args, **kwargs):
        # 强制覆盖：lightning/pyannote 会显式传 weights_only=None/True，
        # setdefault 无效，必须直接赋值。
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)

    _patched._wo_patched = True
    torch.load = _patched


def _preload_cudnn():
    """ctranslate2(faster-whisper 后端)需要在进程里找到 cuDNN 9 / cuBLAS。
    torch 已把它们装在 site-packages/nvidia/*/lib 下，但不在默认链接路径，
    且 torch 以 RTLD_LOCAL 加载、符号不全局。这里用 RTLD_GLOBAL 显式预加载，
    让 ct2 能解析到 cuDNN 符号——与启动方式无关，幂等，失败不致命。"""
    import ctypes
    import glob
    import sysconfig
    if getattr(_preload_cudnn, "_done", False):
        return
    site = sysconfig.get_paths().get("purelib", "")
    nvidia = os.path.join(site, "nvidia")
    if not os.path.isdir(nvidia):
        _preload_cudnn._done = True
        return
    libdirs = [os.path.join(nvidia, "cublas", "lib"),
               os.path.join(nvidia, "cudnn", "lib")]
    # 把目录加进 LD_LIBRARY_PATH（惠及可能的子进程）
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([d for d in libdirs if os.path.isdir(d)] +
                                             ([cur] if cur else []))
    # 收集 so：先 cublas，再 cudnn 子库，最后主库 libcudnn.so.9
    sos = []
    for d in libdirs:
        sos += sorted(glob.glob(os.path.join(d, "*.so*")))
    sos.sort(key=lambda p: (os.path.basename(p).startswith("libcudnn.so"),))
    # 多趟加载以满足相互依赖
    for _ in range(3):
        pending = []
        for so in sos:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pending.append(so)
        if not pending:
            break
        sos = pending
    _preload_cudnn._done = True


def _patch_hf_auth_kwarg():
    """huggingface_hub 1.x 移除了 hf_hub_download 的 use_auth_token 参数（改名 token），
    但 pyannote.audio 3.3.2 仍在传 use_auth_token，导致 diarization 下载模型报
    'unexpected keyword argument use_auth_token'。这里把该参数翻译成 token。
    需要同时改掉 huggingface_hub 模块和已经 `from ... import hf_hub_download` 的引用。"""
    import sys
    import huggingface_hub
    orig = huggingface_hub.hf_hub_download
    if getattr(orig, "_auth_patched", False):
        return

    def wrapped(*args, **kwargs):
        if "use_auth_token" in kwargs:
            tok = kwargs.pop("use_auth_token")
            if "token" not in kwargs:
                kwargs["token"] = tok if isinstance(tok, str) else None
        return orig(*args, **kwargs)

    wrapped._auth_patched = True
    huggingface_hub.hf_hub_download = wrapped
    # 替换所有已导入模块里指向原函数的引用（pyannote 等在导入时已绑定）
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "hf_hub_download", None) is orig:
                setattr(mod, "hf_hub_download", wrapped)
        except Exception:
            pass


def _read_hf_token() -> str:
    """按优先级读取 HuggingFace token：环境变量 → ~/.cache/huggingface/token。
    diarization 用的 pyannote 模型是 gated，必须带 token。"""
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v.strip()
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    return ""


class ASR:
    """WhisperX 转写 + 词级对齐 + pyannote 说话人分离。
    输出 transcript JSON 中每个 segment 多出一个 `speaker` 字段（SPEAKER_00/01…），
    向后兼容旧的 {start,end,text} 结构。"""

    def __init__(self, model_name: str = "medium", language: str = "zh",
                 compute_type: str = None, device: str = None,
                 diarize: bool = True, min_speakers: int = None,
                 max_speakers: int = None, batch_size: int = 16,
                 split_by_speaker: bool = True, speaker_gap: float = 1.0):
        self.logger = None
        self.model_name = model_name
        self.language = language
        self.diarize = diarize
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.batch_size = batch_size
        self.split_by_speaker = split_by_speaker
        self.speaker_gap = speaker_gap

        import torch
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type:
            self.compute_type = compute_type
        else:
            self.compute_type = "float16" if self.device == "cuda" else "int8"

    def set_logger(self, logger):
        self.logger = logger

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def _warn(self, msg):
        if self.logger:
            self.logger.warning(msg)

    def _split_by_speaker(self, seg) -> list:
        """用词级 speaker 把一个转写段拆成若干「单人连续片段」。

        Whisper 的段是按停顿切的，剧情片里一段常有 20+ 秒、跨好几个人的对话；
        whisperx 只把「多数派」speaker 写在段上，词级归属就丢了。拿这种段去算
        声纹，等于用一段混合音代表某个人——这是声纹认不准的源头。
        对齐失败（无 words）时返回 [] 交给调用方按整段落盘。
        """
        if not self.split_by_speaker:
            return []
        words = seg.get("words") or []
        runs = []
        for w in words:
            st, en = w.get("start"), w.get("end")
            txt = str(w.get("word") or "")
            if st is None or en is None:      # 该词没对齐上：并进上一段，不单独成段
                if runs:
                    runs[-1]["text"] += txt
                continue
            spk = w.get("speaker") or "UNKNOWN"
            st, en = float(st), float(en)
            if runs and runs[-1]["speaker"] == spk \
                    and st - runs[-1]["end"] <= self.speaker_gap:
                runs[-1]["end"] = en
                runs[-1]["text"] += txt
            else:
                runs.append({"start": st, "end": en, "speaker": spk, "text": txt})
        return [{"start": round(r["start"], 3), "end": round(r["end"], 3),
                 "text": r["text"].strip(), "speaker": r["speaker"]}
                for r in runs if r["text"].strip()]

    def transcribe(self, video_path: str, output_dir: str = "output") -> Tuple[str, str]:
        import whisperx
        _patch_torch_load()
        _preload_cudnn()

        self._log(f"Starting WhisperX transcription for: {video_path}")
        self._log(f"Device={self.device} compute_type={self.compute_type} "
                  f"model={self.model_name} lang={self.language} diarize={self.diarize}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = Path(video_path).stem

        # ── 1) ASR (faster-whisper backend) ─────────────────────────────
        audio = whisperx.load_audio(video_path)
        asr_model = whisperx.load_model(
            self.model_name, self.device,
            compute_type=self.compute_type, language=self.language,
        )
        self._log("Transcribing (faster-whisper)...")
        result = asr_model.transcribe(audio, batch_size=self.batch_size)
        # 释放 ASR 模型显存，给对齐/分离腾地方
        del asr_model
        gc.collect()
        self._empty_cache()

        # ── 2) 词级对齐（让分离能精确按词归属说话人）──────────────────
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=self.language, device=self.device)
            result = whisperx.align(
                result["segments"], align_model, metadata, audio,
                self.device, return_char_alignments=False)
            del align_model
            gc.collect()
            self._empty_cache()
            self._log("Word-level alignment done")
        except Exception as e:
            self._warn(f"Alignment skipped ({self.language} 可能无对齐模型): {e}")

        # ── 3) 说话人分离（pyannote）────────────────────────────────────
        speaker_assigned = False
        if self.diarize:
            token = _read_hf_token()
            if not token:
                self._warn("未找到 HF token，跳过说话人分离（仅产出无 speaker 的转写）")
            else:
                try:
                    from whisperx.diarize import DiarizationPipeline
                    _patch_hf_auth_kwarg()  # 兼容 huggingface_hub 1.x（pyannote 仍传 use_auth_token）
                    diarize_model = DiarizationPipeline(
                        use_auth_token=token, device=self.device)
                    kw = {}
                    if self.min_speakers:
                        kw["min_speakers"] = self.min_speakers
                    if self.max_speakers:
                        kw["max_speakers"] = self.max_speakers
                    self._log("Running pyannote speaker diarization...")
                    diarize_segments = diarize_model(audio, **kw)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    speaker_assigned = True
                    n_spk = len({s.get("speaker") for s in result["segments"]
                                 if s.get("speaker")})
                    self._log(f"Diarization done: {n_spk} distinct speakers")
                    del diarize_model
                    gc.collect()
                    self._empty_cache()
                except Exception as e:
                    self._warn(f"Diarization failed, 输出无 speaker 转写: {e}")

        # ── 4) 落盘 ─────────────────────────────────────────────────────
        segments = []
        n_split = 0
        for seg in result.get("segments", []):
            parts = self._split_by_speaker(seg) if speaker_assigned else None
            if not parts:
                item = {
                    "start": round(float(seg.get("start", 0.0)), 3),
                    "end": round(float(seg.get("end", 0.0)), 3),
                    "text": (seg.get("text") or "").strip(),
                }
                if speaker_assigned:
                    item["speaker"] = seg.get("speaker") or "UNKNOWN"
                segments.append(item)
                continue
            n_split += len(parts) > 1
            segments.extend(parts)
        if n_split:
            self._log(f"按词级说话人拆分了 {n_split} 个跨人转写段")

        transcript_path = output_path / f"{base_name}_transcript.json"
        srt_path = output_path / f"{base_name}_full.srt"

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
        self._log(f"Transcript saved to: {transcript_path}")

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                spk = f"[{seg['speaker']}] " if seg.get("speaker") else ""
                f.write(f"{i}\n")
                f.write(f"{seconds_to_srt_time(seg['start'])} --> "
                        f"{seconds_to_srt_time(seg['end'])}\n")
                f.write(f"{spk}{seg['text']}\n\n")
        self._log(f"SRT subtitle saved to: {srt_path}")
        self._log(f"Total segments: {len(segments)}")

        return str(transcript_path), str(srt_path)

    def _empty_cache(self):
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass


def transcribe_video(video_path: str, config, output_dir: str = "output") -> Tuple[str, str]:
    asr_cfg = config.asr
    asr = ASR(
        model_name=asr_cfg.get("model", "medium"),
        language=asr_cfg.get("language", "zh"),
        compute_type=asr_cfg.get("compute_type"),
        device=asr_cfg.get("device"),
        diarize=asr_cfg.get("diarize", True),
        min_speakers=asr_cfg.get("min_speakers"),
        max_speakers=asr_cfg.get("max_speakers"),
        batch_size=asr_cfg.get("batch_size", 16),
        split_by_speaker=asr_cfg.get("split_by_speaker", True),
        speaker_gap=float(asr_cfg.get("speaker_gap", 1.0)),
    )
    asr.set_logger(config.get_logger("ASR"))
    return asr.transcribe(video_path, output_dir)


if __name__ == "__main__":
    import sys
    from config import load_config

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    asr = ASR(
        model_name=config.asr.get("model", "medium"),
        language=config.asr.get("language", "zh"),
        diarize=config.asr.get("diarize", True),
    )
    asr.set_logger(config.get_logger("ASR"))

    import glob
    input_dir = config.video.get("input_dir", "input")
    videos = glob.glob(f"{input_dir}/*.mp4") + glob.glob(f"{input_dir}/*.avi")
    if not videos:
        print("No video files found")
        sys.exit(1)
    transcript_path, srt_path = asr.transcribe(videos[0])
    print(f"Transcript: {transcript_path}")
    print(f"SRT: {srt_path}")
