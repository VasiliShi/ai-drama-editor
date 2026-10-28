import json
import logging

from faces import FaceEngine


class _Config:
    def __init__(self, faces=None):
        self.faces = faces or {}
        self.video = {}
        self.scene_notes = {}

    @staticmethod
    def get_logger(_name):
        return logging.getLogger("test.faces")


def test_face_engine_cpu_provider_does_not_probe_cuda():
    engine = FaceEngine(_Config({"device": "cpu"}))

    assert engine._select_providers() == ["CPUExecutionProvider"]


def test_face_engine_auto_falls_back_when_gpu_memory_is_low():
    original = FaceEngine._gpu_free_mb
    FaceEngine._gpu_free_mb = staticmethod(lambda: 1024)
    try:
        engine = FaceEngine(_Config({"device": "auto", "gpu_min_free_mb": 4096}))

        assert engine._select_providers() == ["CPUExecutionProvider"]
    finally:
        FaceEngine._gpu_free_mb = original


def test_face_manifest_can_be_loaded_without_initializing_model(tmp_path):
    manifest = tmp_path / "face_manifest.json"
    manifest.write_text(json.dumps({"clusters": [{"face_id": 7}]}, ensure_ascii=False),
                        encoding="utf-8")
    engine = FaceEngine(_Config({"manifest_file": str(manifest)}))

    assert engine.load_manifest()["clusters"] == [{"face_id": 7}]
    assert engine._app is None


def _unit(v):
    import numpy as np
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_cluster_two_stage_separates_and_attaches():
    import numpy as np
    from faces import cluster_two_stage
    # 两个明显分开的人 + 各自的轻微变体(应吸附) + 一个远离所有人的噪声点
    a = _unit([1.0, 0.0, 0.0])
    a2 = _unit([0.98, 0.20, 0.0])    # 距 a 很近 → 核心簇
    a3 = _unit([0.80, 0.55, 0.0])    # 稍远 → 第二段吸附
    b = _unit([0.0, 1.0, 0.0])
    b2 = _unit([0.15, 0.98, 0.0])
    noise = _unit([0.0, 0.0, 1.0])   # 谁都不像 → -1
    X = np.vstack([a, a2, a3, b, b2, noise])
    labels = cluster_two_stage(X, core_threshold=0.10, attach_threshold=0.45)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert labels[5] == -1


def test_cluster_two_stage_all_singletons_falls_back():
    import numpy as np
    from faces import cluster_two_stage
    X = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
    labels = cluster_two_stage(X, core_threshold=0.05, attach_threshold=0.30)
    # 没有核心簇时退化为单段聚类:无 -1,三点各自成簇
    assert set(labels) == {0, 1, 2}


def test_cluster_two_stage_attaches_profiles_with_looser_threshold():
    import numpy as np
    from faces import cluster_two_stage
    # 一个人:两张正脸(建核心) + 一张侧脸(离质心远,正脸阈值够不着)
    a, a2 = _unit([1.0, 0.05, 0.0]), _unit([1.0, 0.0, 0.05])
    profile = _unit([1.0, 0.85, 0.0])          # 与核心质心距离约 0.24
    b, b2 = _unit([0.0, 0.0, 1.0]), _unit([0.05, 0.0, 1.0])
    X = np.vstack([a, a2, profile, b, b2])
    core_mask = [True, True, False, True, True]

    # 侧脸沿用正脸的紧阈值 → 吸不上,变噪声(这就是改动前的行为)
    tight = cluster_two_stage(X, 0.05, 0.10, core_mask=core_mask,
                              loose_attach_threshold=0.10)
    assert tight[2] == -1
    # 给侧脸更松的阈值 → 正确并入本人的核心簇
    loose = cluster_two_stage(X, 0.05, 0.10, core_mask=core_mask,
                              loose_attach_threshold=0.40)
    assert loose[2] == loose[0] == loose[1]
    assert loose[3] == loose[4] != loose[0]


def test_profile_faces_are_kept_by_query_gate_but_flagged_non_frontal():
    engine = FaceEngine(_Config({"min_eye_ratio": 0.16}))

    # 建库闸严于查询闸,且 min_eye_ratio 只用于标记 frontal,不再作为丢弃条件
    assert engine.query_min_det_score < engine.min_det_score
    assert engine.query_min_face < engine.min_face
    assert engine.query_min_blur < engine.min_blur


def test_match_accepts_low_score_when_it_clearly_leads():
    import numpy as np
    engine = FaceEngine(_Config({"match_threshold": 0.45, "match_floor": 0.30,
                                 "match_margin": 0.06}))
    engine._gallery = [("甲", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                       ("乙", np.array([0.0, 1.0, 0.0], dtype=np.float32))]

    def query(sim_a, sim_b):
        """构造与「甲」余弦=sim_a、与「乙」余弦=sim_b 的单位查询向量。"""
        z = float(np.sqrt(max(0.0, 1 - sim_a ** 2 - sim_b ** 2)))
        return np.array([sim_a, sim_b, z], dtype=np.float32)

    # 正脸:绝对分够高 → 直接认定
    assert engine.match(query(0.90, 0.10))[0] == "甲"
    # 侧脸:0.35 够不到 match_threshold,但比第二名(0.10)领先 0.25 → 认定
    assert engine.match(query(0.35, 0.10))[0] == "甲"
    # 两边都像:领先 0.02 不足 match_margin → 不认,避免张冠李戴
    assert engine.match(query(0.35, 0.33))[0] is None
    # 谁都不像:低于 match_floor → 不认
    assert engine.match(query(0.20, 0.05))[0] is None


def test_farthest_point_sample_covers_spread_not_duplicates():
    import numpy as np
    from faces import farthest_point_sample
    # 三个几乎重复的正脸 + 两个明显不同的角度
    X = np.vstack([_unit([1, 0, 0]), _unit([1, 0.01, 0]), _unit([1, 0.02, 0]),
                   _unit([0, 1, 0]), _unit([0, 0, 1])])
    picked = farthest_point_sample(X, 3)

    assert len(picked) == 3
    assert 3 in picked and 4 in picked      # 两个远角度都被选中,而不是三张重复正脸
