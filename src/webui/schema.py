"""config_schema.yaml 驱动的设置表单：构建控件、收集覆盖、ruamel 回写。

- 表单在 Blocks 构建期一次性循环生成，维护并行的 (components, paths, types)，
  提交时用位置参数 zip 回 path——不依赖 gradio 的 dict-inputs，稳。
- 「保存为默认」用 ruamel round-trip 重读原始 config.yaml 逐项 set 后回写，
  绝不 dump 内存 config（否则 ${ENV} 展开、背景文档内联会被固化）。
"""
import shutil
from pathlib import Path

import gradio as gr
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = REPO_ROOT / "config_schema.yaml"
CONFIG_FILE = REPO_ROOT / "config.yaml"

# 「仅本次生效」的覆盖：模块级，被运行页每次组装 job 时合并（单进程单用户够用）
SESSION_OVERRIDES: dict = {}


def load_schema() -> dict:
    with open(SCHEMA_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_by_path(cfg: dict, path: str):
    node = cfg
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _set_nested(cfg: dict, path: str, value):
    keys = path.split(".")
    node = cfg
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = value


def _coerce(ftype: str, value):
    if ftype == "count":   # 非负整数，0 表示「自动」→ 存 null
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None
    if value is None:
        return None
    if ftype in ("number", "slider"):
        try:
            f = float(value)
            return int(f) if f == int(f) else f
        except (TypeError, ValueError):
            return value
    if ftype == "bool":
        return bool(value)
    return value


def build_form(defaults: dict):
    """按 schema 生成分组 Accordion 表单。返回 (components, paths, types)。"""
    schema = load_schema()
    components, paths, types = [], [], []
    for group in schema.get("groups", []):
        with gr.Accordion(group.get("title", ""), open=False):
            for fld in group.get("fields", []):
                path = fld["path"]
                label = fld.get("label", path)
                ftype = fld.get("type", "text")
                info = fld.get("info")
                cur = get_by_path(defaults, path)
                comp = _make_component(fld, ftype, label, info, cur)
                components.append(comp)
                paths.append(path)
                types.append(ftype)
    return components, paths, types


def _make_component(fld, ftype, label, info, cur):
    if ftype == "bool":
        return gr.Checkbox(value=bool(cur), label=label, info=info)
    if ftype == "choice":
        choices = fld.get("choices", [])
        val = cur if cur in choices else (choices[0] if choices else None)
        return gr.Dropdown(choices=choices, value=val, label=label, info=info,
                           allow_custom_value=True)
    if ftype == "slider":
        return gr.Slider(fld.get("min", 0), fld.get("max", 10),
                         value=float(cur) if cur is not None else fld.get("min", 0),
                         step=fld.get("step", 1), label=label, info=info)
    if ftype == "number":
        return gr.Number(value=cur if cur is not None else 0, label=label,
                         precision=None, info=info)
    if ftype == "count":
        return gr.Number(value=int(cur) if cur else 0, label=label,
                         precision=0, minimum=0, info=info)
    if ftype == "secret":
        tail = str(cur)[-4:] if cur else ""
        ph = f"已设置(尾号 {tail})，留空=不改" if cur else "未设置"
        return gr.Textbox(value="", label=label, info=info, type="password",
                          placeholder=ph)
    return gr.Textbox(value="" if cur is None else str(cur), label=label, info=info)


def collect_overrides(paths, types, values) -> dict:
    """把表单值收集成嵌套 overrides dict。secret 留空表示不改，跳过。"""
    ov = {}
    for path, ftype, val in zip(paths, types, values):
        if ftype == "secret" and not str(val or "").strip():
            continue
        _set_nested(ov, path, _coerce(ftype, val))
    return ov


def apply_once(paths, types, values) -> str:
    global SESSION_OVERRIDES
    SESSION_OVERRIDES = collect_overrides(paths, types, values)
    n = _count_leaves(SESSION_OVERRIDES)
    return f"✅ 已暂存 {n} 项覆盖，仅对之后的运行生效（不写文件）。清空请重启或保存为默认。"


def save_to_yaml(paths, types, values) -> str:
    """ruamel round-trip 回写 config.yaml：重读原文件、逐项 set、写前备份 .bak。"""
    try:
        from ruamel.yaml import YAML
    except ImportError:
        return "❌ 未安装 ruamel.yaml，无法保留注释回写。"
    ov = collect_overrides(paths, types, values)
    yamlrt = YAML(typ="rt")
    yamlrt.preserve_quotes = True
    # 匹配 config.yaml 的块序列缩进（`  - item`：dash 在 2、内容在 4），
    # 否则 round-trip 会把 output_formats 等列表重排、污染无关行。
    yamlrt.indent(mapping=2, sequence=4, offset=2)
    yamlrt.width = 4096  # 不因行宽自动折行长注释
    # 未改动的 null 值保持写成 `null`（默认会渲染成空，污染无关行）
    yamlrt.representer.add_representer(
        type(None),
        lambda r, d: r.represent_scalar("tag:yaml.org,2002:null", "null"))
    with open(CONFIG_FILE, encoding="utf-8") as f:
        doc = yamlrt.load(f)
    changed = _apply_flat(doc, ov)
    if not changed:
        return "（无改动）"
    shutil.copyfile(CONFIG_FILE, str(CONFIG_FILE) + ".bak")
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yamlrt.dump(doc, f)
    return f"✅ 已写入 config.yaml（{len(changed)} 项），备份在 config.yaml.bak：\n" + \
           "\n".join(f"  {p} = {v}" for p, v in changed)


def _apply_flat(doc, ov, prefix=""):
    """把嵌套 ov 逐叶子 set 进 ruamel 文档，返回 [(path, value)] 改动列表。"""
    changed = []
    for k, v in ov.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            if k not in doc or doc[k] is None:
                doc[k] = {}
            changed += _apply_flat(doc[k], v, full)
        else:
            if doc.get(k) != v:
                doc[k] = v
                changed.append((full, v))
    return changed


def _count_leaves(d) -> int:
    n = 0
    for v in d.values():
        n += _count_leaves(v) if isinstance(v, dict) else 1
    return n


def load_yaml_text() -> str:
    return CONFIG_FILE.read_text(encoding="utf-8")


def save_yaml_text(text: str) -> str:
    """高级编辑：整篇 YAML 校验后回写（写前备份）。"""
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as e:
        return f"❌ YAML 语法错误，未保存：{e}"
    shutil.copyfile(CONFIG_FILE, str(CONFIG_FILE) + ".bak")
    CONFIG_FILE.write_text(text, encoding="utf-8")
    return "✅ 已保存整篇 config.yaml（备份在 config.yaml.bak）。"
