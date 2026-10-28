from narration_generator import clamp_beat_end


def test_no_budget_means_unlimited():
    assert clamp_beat_end(10, 15, None) == (15, None)


def test_extension_within_budget():
    end, rem = clamp_beat_end(10, 15, 20.0)
    assert end == 15.0 and rem == 15.0  # 消耗 5s，剩 15s


def test_extension_truncated_by_budget():
    end, rem = clamp_beat_end(10, 18, 3.0)
    assert end == 13.0 and rem == 0.0  # 想延 8s，只准 3s


def test_budget_exhausted_no_extension():
    end, rem = clamp_beat_end(10, 18, 0.0)
    assert end == 10.0 and rem == 0.0


def test_non_extending_beat_keeps_budget():
    # desired <= nat_end：不延长，不消耗预算
    end, rem = clamp_beat_end(10, 8, 5.0)
    assert end == 10.0 and rem == 5.0


def test_cumulative_guard_keeps_total_under_cap():
    """模拟多拍累计：Σ延长量不超过初始预算。"""
    budget = 10.0
    beats = [(0, 5, 12), (0, 20, 30), (0, 40, 50)]  # (i, nat, desired) 各想延 7/10/10
    total_extension = 0.0
    for _i, nat, desired in beats:
        end, budget = clamp_beat_end(nat, desired, budget)
        total_extension += end - nat
    assert budget == 0.0
    assert total_extension == 10.0  # 恰好等于初始预算，不突破
