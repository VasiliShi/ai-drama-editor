"""说话人命名：把转写里的匿名 SPEAKER_xx 换成角色真名——靠读台词，不靠听声音。

为什么不是声纹：身份信号绝大多数直接写在台词里（自称「朕」「奴婢」、称谓「大人」
「干爹」、互相叫名字、报职务「我是河道监管李玄」）。ECAPA 只听音色，对这些完全
无感，在有 BGM/混响/重叠的剧集音频上还极不稳；实测把认不准的段全吸进了戏份最重
的簇（主角簇变垃圾桶），错误名字再一路污染下游。台词是强信号且免费，LLM 读一遍
就能判。

为什么是逐集：pyannote 的 SPEAKER_xx 编号是每集独立分配的，跨集毫无意义（同一个
角色在 ep1 是 SPEAKER_00、ep3 是 SPEAKER_03）。声纹那套全局聚类，唯一目的是把
所有集链接起来好让人只标一次；人一旦不在环里，这个难题就不存在了——每集独立命名，
跨集一致性由「名字都取自同一份角色表」白送。错误也因此被隔离在单集内。

产物：直接改写 transcript，原标签保留在 speaker_raw（便于回查与复算）。
"""
import hashlib
import json
import re
from pathlib import Path

from episode_utils import parse_episode_number

ROSTER_PROMPT = (
    "下面是一部影视作品的背景资料。请抽出其中出现的**角色名单**。\n"
    "只列真正的人物角色名，不要列演员名、地名、机构名、官职本身。\n"
    "同一角色有多种叫法时（本名/字号/尊称），选资料里最常用的那个作为主名。\n\n"
    "【背景资料】\n{background}\n\n"
    '只输出 JSON 数组，如 ["角色A","角色B"]。直接输出 JSON。'
)

NAME_PROMPT = (
    "下面是一集电视剧的对白转写（{lang}），说话人已按声音分离成 SPEAKER_xx，"
    "但编号是随机的，不知道谁是谁。请判断每个 SPEAKER_xx 对应哪个角色。\n\n"
    "判断依据（按可靠度排序）：\n"
    "1. 自称与称谓：如自称「朕」→ 皇帝，自称「奴婢/奴才」→ 太监，"
    "被称「干爹」「大人」「公公」等；\n"
    "2. 直接报出身份或名字：如「我是河道监管李玄」；\n"
    "3. 对话对象与话题：结合角色表判断谁会在这个场合说这些话。\n\n"
    "**没有把握就填空字符串**——保留匿名远好过安错名字，错误的名字会让后续"
    "环节把台词算到别人头上。名字只能从下面的角色表里选，不要发明新名字。\n\n"
    "【角色表】\n{roster}\n\n"
    "【本集对白】\n{dialogue}\n\n"
    '只输出 JSON 数组，每项 {{"speaker":"SPEAKER_xx","name":"角色名或空字符串",'
    '"evidence":"台词里的直接依据，没有就留空"}}。直接输出 JSON。'
)

_LANG_NAMES = {"zh": "中文", "en": "英语", "ja": "日语", "ko": "韩语", "fr": "法语",
               "de": "德语", "es": "西班牙语", "ru": "俄语", "it": "意大利语",
               "pt": "葡萄牙语", "th": "泰语", "vi": "越南语", "ar": "阿拉伯语"}


def lang_name(code) -> str:
    """asr.language 代码 → 中文语种名。别在 prompt 里写死语种：换一部别国的剧，
    写死的「韩语」会直接误导 LLM。"""
    code = str(code or "").strip().lower()
    return _LANG_NAMES.get(code, code or "原声")


def _client(config, logger):
    from llm import LLMClient
    llm = config.llm or {}
    base_url, api_key, model = (llm.get("base_url"), llm.get("api_key"),
                                llm.get("model_name"))
    if not (base_url and api_key and model):
        raise RuntimeError("llm 未配置（base_url/api_key/model_name）")
    # temperature=0：命名是事实抽取，同一份台词必须每次给同样的答案。
    # 服务端默认温度下实测会把关系紧密的角色对调（沈一石↔杨金水）。
    return LLMClient(base_url, api_key, model, logger=logger, temperature=0)


# ── 角色表（每部剧抽一次，按背景文件内容哈希缓存）─────────────────
def load_roster(config, logger=None) -> list:
    """背景资料 → 角色名单。结果缓存到 data/cast_roster.json，背景文件变了才重抽。

    这份名单有两个用途：命名时限定候选（防 LLM 发明角色），以及给评论生成当人名
    白名单（外语剧尤其重要——防同一个角色出现好几种音译写法）。
    """
    background = (config.video or {}).get("series_background") or ""
    if not background.strip():
        if logger:
            logger.warning("video.series_background 为空，无角色表可用")
        return []
    cache = Path((config.naming or {}).get("roster_file", "data/cast_roster.json"))
    key = hashlib.sha1(background.encode("utf-8")).hexdigest()[:16]
    if cache.exists():
        try:
            data = json.load(open(cache, encoding="utf-8"))
            if data.get("key") == key and data.get("names"):
                return list(data["names"])
        except (json.JSONDecodeError, OSError):
            pass
    arr = _client(config, logger).chat_json(
        ROSTER_PROMPT.format(background=background[:20000]))
    names = [str(x).strip() for x in (arr or []) if str(x).strip()]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"key": key, "names": out}, f, ensure_ascii=False, indent=2)
    if logger:
        logger.info(f"角色表: {len(out)} 人 → {cache}")
    return out


# ── 对白摘录（喂给 LLM 的上下文）───────────────────────────────
def build_dialogue(segments, max_chars=24000, n_blocks=12) -> str:
    """转写 → 按时间顺序的「[时间] SPEAKER_xx: 台词」文本。

    保持时间顺序而不是按说话人分组：身份线索多半藏在**互动**里——A 叫 B「大人」、
    B 回「臣…」，拆开分组这些配对就断了。

    超预算时取 n_blocks 段**连续对白**均匀铺满整集，而不是逐行抽稀：
    - 逐行抽稀会把上面那种相邻配对切断，正好毁掉最有价值的信号；
    - 等距抽行还会和对白的周期结构共振（步长恰是说话人轮换周期的整数倍时，
      抽出来全是同一个人）。分块取样对两者都免疫。
    """
    lines = []
    for s in segments:
        spk = s.get("speaker")
        txt = str(s.get("text") or "").strip()
        if spk and txt:
            lines.append(f"[{float(s.get('start', 0)):.0f}s] {spk}: {txt}")
    if not lines:
        return ""
    if sum(len(x) + 1 for x in lines) <= max_chars:
        return "\n".join(lines)

    n_blocks = max(1, min(n_blocks, len(lines)))
    budget = max_chars // n_blocks
    starts = [round(i * (len(lines) - 1) / n_blocks) for i in range(n_blocks)]
    out, taken = [], set()
    for st in starts:
        used = 0
        for j in range(st, len(lines)):
            if j in taken or used + len(lines[j]) + 1 > budget:
                break
            taken.add(j)
            out.append((j, lines[j]))
            used += len(lines[j]) + 1
    return "\n".join(x for _j, x in sorted(out))


# ── 主流程 ─────────────────────────────────────────────────────
def name_speakers(transcript_path: str, config, ep_name: str = None) -> bool:
    """逐集把 SPEAKER_xx 改写成角色真名。返回是否改动了文件。"""
    logger = config.get_logger("Naming")
    path = Path(transcript_path)
    data = json.load(open(path, encoding="utf-8"))
    segments = data.get("segments", [])
    speakers = {s.get("speaker") for s in segments if s.get("speaker")}
    speakers = {s for s in speakers if re.fullmatch(r"SPEAKER_\d+", str(s))}
    if not speakers:
        logger.info("转写里没有待命名的 SPEAKER_xx，跳过")
        return False

    roster = load_roster(config, logger)
    if not roster:
        logger.warning("没有角色表，跳过命名（请填 video.series_background_file）")
        return False

    dialogue = build_dialogue(
        segments, int((config.naming or {}).get("max_dialogue_chars", 24000)))
    if not dialogue:
        return False
    prompt = NAME_PROMPT.format(
        lang=lang_name((config.asr or {}).get("language")),
        roster="、".join(roster), dialogue=dialogue)
    arr = _client(config, logger).chat_json(prompt)

    allowed = set(roster)
    mapping, evidence = {}, {}
    for item in (arr or []):
        if not isinstance(item, dict):
            continue
        spk = str(item.get("speaker") or "").strip()
        name = str(item.get("name") or "").strip()
        if spk not in speakers or not name:
            continue
        if name not in allowed:          # 白名单外的名字一律丢弃，宁可保持匿名
            logger.warning(f"{spk} 的猜测「{name}」不在角色表里，忽略")
            continue
        mapping[spk] = name
        evidence[spk] = str(item.get("evidence") or "").strip()

    if not mapping:
        logger.warning("LLM 没能确定任何说话人，全部保持匿名")
        return False

    renamed = 0
    for s in segments:
        spk = s.get("speaker")
        name = mapping.get(spk)
        if name:
            s.setdefault("speaker_raw", spk)
            s["speaker"] = name
            renamed += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    ep = ep_name or parse_episode_number(path.stem)
    logger.info(f"ep{ep}: {len(mapping)}/{len(speakers)} 个说话人命名成功，"
                f"改写 {renamed} 段")
    for spk, name in sorted(mapping.items()):
        logger.info(f"    {spk} → {name}"
                    + (f"（依据：{evidence[spk][:40]}）" if evidence.get(spk) else ""))
    return True


def revert_names(transcript_path: str) -> int:
    """把 speaker_raw 写回 speaker，撤销上一次命名。返回还原的段数。

    命名过的转写里已经没有 SPEAKER_xx，重跑命名会直接跳过——想重命名（例如从旧的
    声纹命名迁移过来、或改了背景角色表）必须先还原。
    """
    path = Path(transcript_path)
    data = json.load(open(path, encoding="utf-8"))
    n = 0
    for s in data.get("segments", []):
        if "speaker_raw" in s:
            s["speaker"] = s.pop("speaker_raw")
            n += 1
    if n:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return n


def main():
    import argparse
    import logging
    from config import load_config

    ap = argparse.ArgumentParser(
        description="说话人命名：LLM 读台词把 SPEAKER_xx 换成角色真名")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output-base", default="output")
    ap.add_argument("--episode", help="只处理某一集（目录名），默认全部")
    ap.add_argument("--revert", action="store_true",
                    help="只还原成 SPEAKER_xx，不重新命名")
    ap.add_argument("--rename", action="store_true",
                    help="已命名的转写先还原再重新命名（改了角色表/从旧方案迁移时用）")
    args = ap.parse_args()

    config = load_config(args.config)
    logging.getLogger("VideoEditor.Naming").setLevel(logging.INFO)
    base = Path(args.output_base)
    eps = ([base / args.episode] if args.episode
           else sorted((d for d in base.glob("*") if d.is_dir()),
                       key=lambda d: (not d.name.isdigit(), d.name)))
    done = 0
    for d in eps:
        tp = d / f"{d.name}_transcript.json"
        if not tp.exists():
            continue
        if args.revert or args.rename:
            n = revert_names(str(tp))
            print(f"ep{d.name}: 还原 {n} 段", flush=True)
            if args.revert:
                continue
        try:
            if name_speakers(str(tp), config, d.name):
                done += 1
        except Exception as exc:
            print(f"ep{d.name}: 命名失败 {exc}", flush=True)
    print(f"完成：{done} 集已命名", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
