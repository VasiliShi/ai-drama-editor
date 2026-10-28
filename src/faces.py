"""人脸库：人脸先行标注 + 运行期确定性认人。

分工（与 VLM 的边界）：
- 本模块(InsightFace/ArcFace)只回答「这张脸是谁」——检测、embedding、跨帧同人比对；
- Qwen-VL 只负责「在做什么/什么场景」；画面备注把本模块认出的名字作为已知事实
  写进 VLM prompt，VLM 不再隔空对脸。

流程：
  1) prepare(): 各集转写的说话时刻抽帧 → 检测人脸 → 裁剪+embedding → 阈值聚类
     → face_manifest.json（按出场次数排序的人物候选，每人一组脸缩略图）。
  2) webui「人脸标注」页看脸打名字 → build_gallery() 写 face_gallery.json
     （名字→质心向量），并把最清晰的一张存 data/cast/真名.jpg（VLM 参考图）。
  3) identify_frame(): 画面备注(VLM)的在场人物用它确定性认人。
"""
import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from media import run_ffmpeg


def farthest_point_sample(X, k):
    """贪心最远点采样,返回 k 个互相最不像的成员下标。

    用途:一个人的 exemplar 集要覆盖不同角度(正脸/左右侧脸/低头)。
    取「离已选集合最远」的成员,比取「离质心最近」的成员覆盖面大得多——
    后者选出来的全是同一个正脸角度,正是侧脸认不出的原因之一。
    """
    n = len(X)
    if n == 0:
        return []
    k = min(k, n)
    picked = [0]
    dist = 1.0 - (X @ X[0])          # dist[i] = 1 - max_{j∈picked} cos(X[i], X[j])
    while len(picked) < k:
        i = int(np.argmax(dist))
        if dist[i] <= 1e-6:          # 剩下的都和已选集合重复
            break
        picked.append(i)
        dist = np.minimum(dist, 1.0 - (X @ X[i]))
    return picked


def cluster_two_stage(X, core_threshold=0.35, attach_threshold=0.60,
                      core_mask=None, loose_attach_threshold=None):
    """两段式聚类:先紧后松,治 average-linkage 单一阈值的链式混人与碎簇。

    第一段:只用「可信正脸」(core_mask)以紧阈值(cosine 距离 core_threshold)聚出
      "确定同人"核心簇(≥2 张)。侧脸/糊脸 embedding 噪声大,不参与建核心,
      否则会把两个人链到一起。
    第二段:其余脸逐个吸附到最近核心簇。正脸用 attach_threshold,非正脸用
      loose_attach_threshold(更松)——同人跨姿态的余弦本来就低,用同一把尺子量
      必然把侧脸判成噪声,这正是「同一人不同侧脸识别失败」的根因。
    返回 labels(np.ndarray, -1=噪声)。X 需已 L2 归一化。
    """
    from sklearn.cluster import AgglomerativeClustering
    n = len(X)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    if loose_attach_threshold is None:
        loose_attach_threshold = attach_threshold
    core_mask = (np.ones(n, dtype=bool) if core_mask is None
                 else np.asarray(core_mask, dtype=bool))
    core_idx = np.flatnonzero(core_mask)
    if len(core_idx) < 2:      # 正脸太少建不起核心:退化为全量单段松阈值
        return AgglomerativeClustering(
            n_clusters=None, distance_threshold=attach_threshold,
            metric="cosine", linkage="average").fit_predict(X)

    tight = AgglomerativeClustering(
        n_clusters=None, distance_threshold=core_threshold,
        metric="cosine", linkage="average").fit_predict(X[core_idx])
    sizes = {}
    for lab in tight:
        sizes[int(lab)] = sizes.get(int(lab), 0) + 1
    core_labs = sorted(lab for lab, sz in sizes.items() if sz >= 2)
    if not core_labs:
        return AgglomerativeClustering(
            n_clusters=None, distance_threshold=attach_threshold,
            metric="cosine", linkage="average").fit_predict(X)
    remap = {lab: i for i, lab in enumerate(core_labs)}
    centroids = np.vstack([
        (lambda c: c / (np.linalg.norm(c) or 1))(X[core_idx[tight == lab]].mean(axis=0))
        for lab in core_labs])

    labels = np.full(n, -1, dtype=int)
    for pos, i in enumerate(core_idx):        # 核心簇成员直接落位
        lab = int(tight[pos])
        if lab in remap:
            labels[i] = remap[lab]
    for i in range(n):                        # 其余(核心散脸 + 全部非正脸)吸附
        if labels[i] >= 0:
            continue
        limit = attach_threshold if core_mask[i] else loose_attach_threshold
        sims = centroids @ X[i]
        j = int(np.argmax(sims))
        if 1 - float(sims[j]) <= limit:
            labels[i] = j
    return labels


class FaceEngine:
    def __init__(self, config):
        self.config = config
        self.cfg = getattr(config, "faces", {}) or {}
        self.video_cfg = config.video or {}
        self.logger = config.get_logger("Faces")

        self.det_model = self.cfg.get("det_model", "buffalo_l")
        self.sample_dir = Path(self.cfg.get("sample_dir", "output/faces"))
        self.manifest_file = self.cfg.get("manifest_file", "data/face_manifest.json")
        self.gallery_file = self.cfg.get("gallery_file", "data/face_gallery.json")
        self.cluster_threshold = float(self.cfg.get("cluster_threshold", 0.60))
        self.core_threshold = float(self.cfg.get("core_threshold", 0.35))
        self.profile_attach_threshold = float(
            self.cfg.get("profile_attach_threshold", 0.72))
        self.match_threshold = float(self.cfg.get("match_threshold", 0.45))
        self.match_floor = float(self.cfg.get("match_floor", 0.30))
        self.match_margin = float(self.cfg.get("match_margin", 0.06))
        self.exemplars_per_person = int(self.cfg.get("exemplars_per_person", 8))
        self.flip_tta = bool(self.cfg.get("flip_tta", True))
        self.outlier_sim = float(self.cfg.get("outlier_sim", 0.30))
        self.profile_outlier_sim = float(self.cfg.get("profile_outlier_sim", 0.20))
        # 建库闸门(严):只有过闸的脸才配当聚类核心
        self.min_face = int(self.cfg.get("min_face", 60))
        self.min_det_score = float(self.cfg.get("min_det_score", 0.50))
        self.min_blur = float(self.cfg.get("min_blur", 20.0))
        self.min_eye_ratio = float(self.cfg.get("min_eye_ratio", 0.16))
        # 运行期闸门(松):侧脸/稍糊也要认,否则等于主动放弃识别
        self.query_min_face = int(self.cfg.get("query_min_face", 40))
        self.query_min_det_score = float(self.cfg.get("query_min_det_score", 0.40))
        self.query_min_blur = float(self.cfg.get("query_min_blur", 8.0))
        self.frames_per_speaker = int(self.cfg.get("frames_per_speaker", 12))
        self.num_candidates = int(self.cfg.get("num_candidates", 20))
        self._on_gpu = False
        self._cpu_affinity = None
        self.max_episodes = int(self.cfg.get("max_episodes", 4))
        self.device = str(self.cfg.get("device", "auto") or "auto").strip().lower()
        if self.device not in {"auto", "cpu", "cuda"}:
            self.logger.warning(f"未知 faces.device={self.device!r}，按 auto 处理")
            self.device = "auto"
        self.gpu_min_free_mb = max(0, int(self.cfg.get("gpu_min_free_mb", 4096) or 0))
        self.cpu_threads = max(1, int(self.cfg.get("cpu_threads", 2) or 1))
        self.max_frames = max(1, int(self.cfg.get("max_frames", 96) or 1))
        self.max_faces_per_frame = max(1, int(self.cfg.get("max_faces_per_frame", 8) or 1))
        self.det_size = max(160, int(self.cfg.get("det_size", 640) or 640))
        self.cast_dir = Path(config.scene_notes.get("cast_dir", "data/cast") or "data/cast")
        self._app = None
        self._gallery = None   # [(name, centroid np.ndarray)]

    # ── 模型 ──────────────────────────────────────────────────────
    @staticmethod
    def _gpu_free_mb():
        """返回 GPU 0 的可用显存；无法探测时返回 None。"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode != 0:
                return None
            return int(result.stdout.strip().splitlines()[0].strip())
        except (IndexError, OSError, subprocess.SubprocessError, ValueError):
            return None

    def _select_providers(self):
        """避免人脸标注与 Notebook/VLM 无条件争抢 GPU。"""
        if self.device == "cpu":
            self.logger.info("InsightFace device=cpu（faces.device）")
            return ["CPUExecutionProvider"]
        if self.device == "auto":
            free_mb = self._gpu_free_mb()
            if free_mb is None:
                self.logger.warning("无法探测 GPU 空闲显存，InsightFace 回退 CPU")
                return ["CPUExecutionProvider"]
            if free_mb < self.gpu_min_free_mb:
                self.logger.warning(
                    f"GPU 空闲显存 {free_mb}MB < faces.gpu_min_free_mb "
                    f"{self.gpu_min_free_mb}MB，InsightFace 回退 CPU")
                return ["CPUExecutionProvider"]
            self.logger.info(f"GPU 空闲显存 {free_mb}MB，InsightFace 使用 CUDA")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _load_app(self, providers=None):
        if self._app is not None:
            return
        providers = providers or self._select_providers()
        wants_gpu = providers[0] == "CUDAExecutionProvider"
        if not wants_gpu and self._cpu_affinity is None:
            self._cpu_affinity = self._limit_cpu()
        # 复用 asr 的 cudnn 预加载：onnxruntime-gpu 的 CUDA provider 需要 cudnn/cublas，
        # 系统没装全局 CUDA 库，用 torch 自带的（dlopen 预载后 provider 才初始化得起来）
        if wants_gpu:
            try:
                from asr import _preload_cudnn
                _preload_cudnn()
            except Exception:
                pass
        from insightface.app import FaceAnalysis
        self.logger.info(f"Loading InsightFace ({self.det_model})")
        try:
            self._app = FaceAnalysis(name=self.det_model, root="data/insightface",
                                     providers=providers)
            self._app.prepare(ctx_id=0 if wants_gpu else -1,
                              det_size=(self.det_size, self.det_size))
        except Exception:
            if not wants_gpu or self.device == "cuda":
                raise
            self.logger.exception("InsightFace CUDA 初始化失败，回退 CPU")
            if self._cpu_affinity is None:
                self._cpu_affinity = self._limit_cpu()
            self._app = FaceAnalysis(name=self.det_model, root="data/insightface",
                                     providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
        provs = self._app.models["detection"].session.get_providers()
        self._on_gpu = "CUDAExecutionProvider" in provs
        self.logger.info(f"InsightFace providers: {provs} (GPU={self._on_gpu})")

    def _limit_cpu(self):
        """CPU 回退时固定少量核心，避免 ONNX Runtime 吃满宿主机。"""
        try:
            cores = sorted(os.sched_getaffinity(0))
            limit = min(self.cpu_threads, len(cores))
            os.sched_setaffinity(0, set(cores[:limit]))
            self.logger.info(f"CPU 模式：限用 {limit}/{len(cores)} 核")
            return set(cores)
        except Exception:
            return None

    @staticmethod
    def _restore_cpu(old):
        if old:
            try:
                os.sched_setaffinity(0, old)
            except Exception:
                pass

    def _embed(self, img, face):
        """脸 → 单位化 embedding。flip_tta 时与镜像图的 embedding 平均。

        ArcFace 近似翻转等变:左侧脸镜像后长得像右侧脸,两者平均得到的向量对
        姿态更不敏感——跨角度余弦能抬 0.03~0.08,正是侧脸认人差的那一档。
        建库与查询必须都开或都关,否则两侧向量不在同一空间。
        """
        vec = np.asarray(face.normed_embedding, dtype=np.float32)
        if not self.flip_tta:
            return vec
        try:
            import cv2
            from insightface.utils import face_align
            rec = self._app.models.get("recognition")
            kps = getattr(face, "kps", None)
            if rec is None or kps is None:
                return vec
            aimg = face_align.norm_crop(img, landmark=kps, image_size=112)
            feat = np.asarray(rec.get_feat(cv2.flip(aimg, 1))).flatten()
            feat = feat / (np.linalg.norm(feat) or 1)
            vec = vec + feat
            return vec / (np.linalg.norm(vec) or 1)
        except Exception:
            return vec

    def _detect(self, img_path: str, mode: str = "enroll"):
        """返回 [{bbox, score, vec, frontal, eye_ratio}]，按 面积×置信度 降序。

        两套闸门,分别服务两个目的:
        - enroll(建库):严。糊脸/低置信的 embedding 噪声大,是聚类混人和碎簇的主因,
          尺寸 ≥min_face、置信 ≥min_det_score、清晰度 ≥min_blur 才收。
        - query(运行期认人):松。此处只想「尽量把脸交给匹配器」,由 match() 的
          阈值+margin 决定认不认；在检测阶段就丢脸等于主动放弃识别。
        眼距/框宽(min_eye_ratio)不再作为丢弃条件,只标记 frontal——侧脸照收,
        但不让它参与聚类建核心。
        """
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        strict = mode == "enroll"
        min_face = self.min_face if strict else self.query_min_face
        min_score = self.min_det_score if strict else self.query_min_det_score
        min_blur = self.min_blur if strict else self.query_min_blur
        out = []
        for f in self._app.get(img):
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            w, h = x2 - x1, y2 - y1
            if min(w, h) < min_face or float(f.det_score) < min_score:
                continue
            kps = getattr(f, "kps", None)
            eye_ratio = (abs(float(kps[1][0]) - float(kps[0][0])) / w
                         if kps is not None and w > 0 else 1.0)
            roi = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            if roi.size:
                blur = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                if blur < min_blur:
                    continue
            out.append({"bbox": (x1, y1, x2, y2), "score": float(f.det_score),
                        "vec": self._embed(img, f), "eye_ratio": eye_ratio,
                        "frontal": eye_ratio >= self.min_eye_ratio})
        # 说话人基本是画面里最大的脸:按 面积×置信度 排序,截断时先保大脸
        out.sort(key=lambda d: ((d["bbox"][2] - d["bbox"][0])
                                * (d["bbox"][3] - d["bbox"][1])) * d["score"],
                 reverse=True)
        return out

    @staticmethod
    def _crop(img_path: str, bbox, out_path: str, margin: float = 0.4):
        """人脸框外扩一圈裁剪（带头发/肩膀，比纯脸更利于人认与比对）。"""
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        mw, mh = int((x2 - x1) * margin), int((y2 - y1) * margin)
        x1, y1 = max(0, x1 - mw), max(0, y1 - int(mh * 1.5))  # 头顶多留
        x2, y2 = min(w, x2 + mw), min(h, y2 + mh)
        cv2.imwrite(str(out_path), img[y1:y2, x1:x2])
        return True

    # ── 1) 抽帧→检脸→聚类→候选清单 ─────────────────────────────────
    def _find_episodes(self, output_base="output"):
        """返回 [(ep_name, transcript_path, video_path)]，仅含带 speaker 的转写。"""
        input_dir = Path(self.video_cfg.get("input_dir", "input"))
        exts = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv")
        episodes = []
        for d in sorted(Path(output_base).glob("*")):
            if not d.is_dir():
                continue
            tp = d / f"{d.name}_transcript.json"
            if not tp.exists():
                continue
            try:
                segs = json.load(open(tp, encoding="utf-8")).get("segments", [])
            except (json.JSONDecodeError, OSError):
                continue
            if not any(s.get("speaker") for s in segs):
                continue
            vp = next((input_dir / f"{d.name}{e}" for e in exts
                       if (input_dir / f"{d.name}{e}").exists()), None)
            if vp is None:
                self.logger.warning(f"ep{d.name}: 找不到源视频，跳过")
                continue
            episodes.append((d.name, str(tp), str(vp)))
        return episodes

    def prepare(self, output_base="output") -> str:
        episodes = self._find_episodes(output_base)
        if not episodes:
            raise RuntimeError("没有带 speaker 的转写，请先「仅转写」各集")
        # 人脸库不需要全剧：主角前几集就全亮过相，建库集数封顶(几十集也不会跑满机器)
        if len(episodes) > self.max_episodes:
            episodes = episodes[: self.max_episodes]
            self.logger.info(f"建库只用前 {self.max_episodes} 集(faces.max_episodes)")

        providers = self._select_providers()
        if providers[0] == "CPUExecutionProvider":
            self._cpu_affinity = self._limit_cpu()
        frames_dir = self.sample_dir / "frames"
        crops_dir = self.sample_dir / "crops"
        frames_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._load_app(providers)
            # 说话时刻抽帧（每个(集,speaker)取最长的 N 段中点）→ 检脸
            faces = []   # {ep, t, frame, crop, score, vec}
            sampled_frames = 0
            for ep_name, tp, vp in episodes:
                remaining = self.max_frames - sampled_frames
                if remaining <= 0:
                    self.logger.info(f"抽帧已达到 faces.max_frames={self.max_frames}，跳过后续集")
                    break
                segs = json.load(open(tp, encoding="utf-8")).get("segments", [])
                by_spk = {}
                for s in segs:
                    spk = s.get("speaker")
                    dur = float(s.get("end", 0)) - float(s.get("start", 0))
                    if spk and spk != "UNKNOWN" and dur >= 1.0:
                        by_spk.setdefault(spk, []).append((float(s["start"]), dur))

                ranked_by_speaker = {
                    spk: sorted(slist, key=lambda item: item[1], reverse=True)
                    [:self.frames_per_speaker]
                    for spk, slist in by_spk.items()
                }
                samples = []
                rounds = max((len(slist) for slist in ranked_by_speaker.values()), default=0)
                for index in range(rounds):
                    choices = sorted(
                        ((spk, slist[index][0], slist[index][1])
                         for spk, slist in ranked_by_speaker.items() if index < len(slist)),
                        key=lambda item: (-item[2], str(item[0]), item[1]),
                    )
                    samples.extend(choices)
                    if len(samples) >= remaining:
                        break
                samples = samples[:remaining]

                n_before = len(faces)
                for _spk, st, dur in samples:
                    t = st + dur / 2
                    frame = frames_dir / f"{ep_name}_{t:.1f}.jpg"
                    if not frame.exists():
                        run_ffmpeg(["-threads", str(self.cpu_threads), "-ss", f"{t:.2f}",
                                    "-i", vp, "-frames:v", "1", "-vf", "scale=960:-2",
                                    "-q:v", "3", str(frame)], check=False)
                    sampled_frames += 1
                    if not frame.exists():
                        continue
                    detected = self._detect(frame, mode="enroll")  # 已按 面积×置信度 排序
                    for k, d in enumerate(detected[:self.max_faces_per_frame]):
                        crop = crops_dir / f"{ep_name}_{t:.1f}_{k}.jpg"
                        if self._crop(frame, d["bbox"], crop):
                            faces.append({"ep": ep_name, "t": t, "frame": str(frame),
                                          "crop": str(crop), "score": d["score"],
                                          "vec": d["vec"], "frontal": d["frontal"]})
                self.logger.info(
                    f"ep{ep_name}: 抽 {len(samples)} 帧，检出 {len(faces) - n_before} 张脸")

            if not faces:
                raise RuntimeError("没有检测到人脸")

            # 已标注人物作锚点:认识的脸先吸走(增量标注,重新生成不用重填名字)
            anchored = {}   # name -> [face]
            if self.load_gallery():
                rest = []
                for f in faces:
                    name, sim = self.match(f["vec"])
                    if name:
                        anchored.setdefault(name, []).append(f)
                    else:
                        rest.append(f)
                if anchored:
                    self.logger.info(
                        f"锚点吸附: {sum(len(v) for v in anchored.values())} 张脸 → "
                        f"{'、'.join(anchored)} (剩 {len(rest)} 张待聚类)")
                faces = rest

            # 两段式聚类:正脸建核心(core),其余含侧脸按更松阈值吸附(attach)
            groups = []   # (name_or_empty, [face])
            if faces:
                X = np.vstack([f["vec"] for f in faces])
                core_mask = np.array([f.get("frontal", True) for f in faces], dtype=bool)
                n_profile = int((~core_mask).sum())
                if n_profile:
                    self.logger.info(
                        f"侧脸 {n_profile}/{len(faces)} 张:不参与建核心,按 "
                        f"profile_attach_threshold={self.profile_attach_threshold} 吸附")
                labels = cluster_two_stage(
                    X, self.core_threshold, self.cluster_threshold,
                    core_mask=core_mask,
                    loose_attach_threshold=self.profile_attach_threshold)
                by_label = {}
                n_noise = 0
                for f, lab in zip(faces, labels):
                    if lab < 0:
                        n_noise += 1
                        continue
                    by_label.setdefault(int(lab), []).append(f)
                if n_noise:
                    self.logger.info(f"聚类丢弃 {n_noise} 张孤立噪声脸")
                groups += [("", flist) for flist in by_label.values()]
            groups += [(name, flist) for name, flist in anchored.items()]

            cand = []
            # 已命名锚点簇排最前(方便核对),其余按簇大小
            groups.sort(key=lambda g: (g[0] == "", -len(g[1])))
            for fid, (name, flist) in enumerate(groups[:self.num_candidates]):
                # 质心只用正脸算:侧脸向量会把质心拽偏,反过来又误伤正脸成员
                fronts = [f for f in flist if f.get("frontal", True)] or flist
                centroid = np.mean([f["vec"] for f in fronts], axis=0)
                centroid = centroid / (np.linalg.norm(centroid) or 1)
                # 簇内提纯:离质心过远的成员踢出(多为误吸附)。侧脸用更松的门槛,
                # 否则刚吸附进来的侧脸立刻又被当离群剔掉,exemplar 集退化回单一角度
                kept = []
                for f in flist:
                    s = float(np.dot(f["vec"], centroid))
                    floor = (self.outlier_sim if f.get("frontal", True)
                             else self.profile_outlier_sim)
                    if s >= floor:
                        kept.append((f, s))
                if not kept:
                    kept = [(f, float(np.dot(f["vec"], centroid))) for f in flist]
                front_sims = [s for f, s in kept if f.get("frontal", True)]
                purity = float(np.mean(front_sims or [s for _, s in kept]))
                # 展示：优先清晰(det_score 高)且离质心近的脸
                ranked = sorted((f for f, _ in kept),
                                key=lambda f: (float(np.dot(f["vec"], centroid))
                                               + f["score"]), reverse=True)
                # 多姿态 exemplar:质心 + 最远点采样的成员。运行期取 max 相似度,
                # 单一质心模板认不出的侧脸,由侧脸 exemplar 直接命中
                M = np.vstack([f["vec"] for f, _ in kept])
                ex_idx = farthest_point_sample(M, max(1, self.exemplars_per_person - 1))
                exemplars = [centroid] + [M[i] for i in ex_idx]
                cand.append({
                    "face_id": fid,
                    "count": len(kept),
                    "n_profile": sum(1 for f, _ in kept if not f.get("frontal", True)),
                    "episodes": sorted({f["ep"] for f, _ in kept}),
                    "crops": [f["crop"] for f in ranked[:9]],
                    "frames": [f["frame"] for f in ranked[:9]],
                    "best_crop": ranked[0]["crop"],
                    "centroid": [round(float(x), 5) for x in centroid],
                    "exemplars": [[round(float(x), 5) for x in v] for v in exemplars],
                    "purity": round(purity, 3),
                    "name": name,
                })

            manifest = {"clusters": cand}
            Path(self.manifest_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Face manifest: {len(groups)} clusters"
                             f"(含锚点 {len(anchored)}), "
                             f"导出前 {len(cand)} 个人物候选 → {self.manifest_file}")
            return self.manifest_file
        finally:
            self._restore_cpu(self._cpu_affinity)
            self._cpu_affinity = None

    def close(self):
        """显式释放 ORT 会话；候选生成子进程退出前调用。"""
        self._app = None
        gc.collect()
        if self._on_gpu:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        self._on_gpu = False
        self._restore_cpu(self._cpu_affinity)
        self._cpu_affinity = None
    def load_manifest(self) -> dict:
        if Path(self.manifest_file).exists():
            return json.load(open(self.manifest_file, encoding="utf-8"))
        return {}

    # ── 2) 建人脸库 + cast 参考图 ───────────────────────────────────
    def build_gallery(self, name_map: dict) -> str:
        """name_map: {face_id: 真名}。同名多簇→多姿态 exemplar 并存（都算这个人）。

        存的是 exemplar 集(质心 + 最远点采样的成员)而不是单个质心:一个人的正脸和
        侧脸在 ArcFace 空间里离得很远,单峰模板对侧脸必然失配。
        """
        manifest = self.load_manifest()
        by_id = {c["face_id"]: c for c in manifest.get("clusters", [])}
        gallery = {}
        import shutil
        for fid, name in (name_map or {}).items():
            name = str(name or "").strip()
            c = by_id.get(int(fid))
            if not name or not c:
                continue
            gallery.setdefault(name, []).extend(
                c.get("exemplars") or [c["centroid"]])
            dst = self.cast_dir / f"{name}.jpg"
            if not dst.exists() and Path(c["best_crop"]).exists():
                self.cast_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(c["best_crop"], dst)
        n_ex = sum(len(v) for v in gallery.values())
        # flip_tta 改变 embedding 空间，旧库与新查询不可比 → 记版本，load 时校验
        gallery["_meta"] = {"flip_tta": self.flip_tta, "format": "exemplars"}
        Path(self.gallery_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.gallery_file, "w", encoding="utf-8") as f:
            json.dump(gallery, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Face gallery: {len(gallery) - 1} 人 / {n_ex} exemplars "
                         f"→ {self.gallery_file}")
        return self.gallery_file

    # ── 3) 运行期认人 ───────────────────────────────────────────────
    def load_gallery(self) -> bool:
        if self._gallery is not None:
            return bool(self._gallery)
        self._gallery = []
        if Path(self.gallery_file).exists():
            data = json.load(open(self.gallery_file, encoding="utf-8"))
            meta = data.get("_meta") or {}
            for name, cents in data.items():
                if name.startswith("_"):
                    continue
                for c in cents:
                    v = np.asarray(c, dtype=np.float32)
                    self._gallery.append((name, v / (np.linalg.norm(v) or 1)))
            if self._gallery and meta.get("flip_tta") != self.flip_tta:
                self.logger.warning(
                    f"人脸库是用 flip_tta={meta.get('flip_tta')} 建的，当前 "
                    f"flip_tta={self.flip_tta}，两边 embedding 不在同一空间，"
                    f"识别会明显变差 → 请在「人脸标注」页重新生成候选并保存重建库")
        return bool(self._gallery)

    def match(self, vec) -> tuple:
        """embedding → (名字, 相似度)；认不出返回 (None, best_sim)。

        每人取其 exemplar 的最大相似度(不是质心相似度),再用两条判据之一放行:
        - 绝对高分:best ≥ match_threshold,正脸走这条;
        - 相对领先:best ≥ match_floor 且比第二名高出 match_margin。侧脸的绝对分
          天然偏低,但只要它明显更像 A 而不是 B,就该判 A。
        单纯调低阈值会张冠李戴、调高会认不出,margin 判据才是侧脸那一档的解。
        """
        if not self.load_gallery():
            return None, 0.0
        best = {}
        for name, g in self._gallery:
            s = float(np.dot(vec, g))
            if s > best.get(name, -1.0):
                best[name] = s
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        name, sim = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else -1.0
        if sim >= self.match_threshold:
            return name, sim
        if sim >= self.match_floor and (sim - second) >= self.match_margin:
            return name, sim
        return None, sim

    def identify_frame(self, img_path: str) -> list:
        """整帧 → 认出的名字列表（去重，未入库的脸忽略）。"""
        if not self.load_gallery():
            return []
        self._load_app()
        names = []
        for d in self._detect(img_path, mode="query"):
            name, _sim = self.match(d["vec"])
            if name and name not in names:
                names.append(name)
        return names


_ENGINE_CACHE = {}


def get_engine(config):
    """进程内复用 FaceEngine：InsightFace 会话加载要数秒，逐帧新建等于每帧重载模型。"""
    key = id(config)
    eng = _ENGINE_CACHE.get(key)
    if eng is None:
        eng = FaceEngine(config)
        _ENGINE_CACHE[key] = eng
    return eng


def main():
    parser = argparse.ArgumentParser(description="生成人脸标注候选")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    from config import load_config

    engine = None
    try:
        engine = FaceEngine(load_config(args.config))
        manifest = engine.prepare()
        print(f"人脸候选已生成：{manifest}", flush=True)
        return 0
    except Exception as exc:
        print(f"生成人脸候选失败：{exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
