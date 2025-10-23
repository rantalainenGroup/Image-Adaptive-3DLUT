# utils/config.py
import argparse, datetime, json, shutil, sys
from pathlib import Path
from typing import Dict, Any, Tuple, Set, Optional, List

# ---------- File I/O ----------

def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML or JSON config. Return {} if path is None."""
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    sfx = p.suffix.lower()
    if sfx in (".yml", ".yaml"):
        try:
            import yaml  # pip install pyyaml
        except ImportError as e:
            raise ImportError("YAML config requires PyYAML. Install: pip install pyyaml") from e
        return yaml.safe_load(text) or {}
    if sfx == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config extension: {sfx} (use .yaml/.yml/.json)")

def dump_human_config(data: Dict[str, Any], dest: Path) -> None:
    """Write YAML if available, else JSON; both human-friendly."""
    try:
        import yaml
        dest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception:
        dest.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")

# ---------- Run directory & snapshots ----------

def make_run_dir(output_dir: str, run_name: Optional[str]) -> Path:
    root = Path(output_dir).expanduser().resolve()
    if not run_name or run_name.lower() == "auto":
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"saved_models_{stamp}"
    else:
        run_dir = root / run_name / "saved_models"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_configs(run_dir: Path, effective_cfg: Dict[str, Any], original_cfg_path: Optional[str]) -> None:
    """Copy original config (if any) and write effective merged config."""
    if original_cfg_path:
        src = Path(original_cfg_path).expanduser()
        if src.exists():
            shutil.copy2(src, run_dir / f"config_original{src.suffix}")
    dump_human_config(effective_cfg, run_dir / "config_effective.yaml")

# ---------- Core: Option 1 merging (defaults < file < CLI-provided) ----------

def _discover_cli_provided_dests(parser: argparse.ArgumentParser, argv: List[str]) -> Set[str]:
    """
    Return dest names for options that actually appeared in argv.
    Handles '--opt=val', '--opt val', and short '-o val'.
    """
    actions = getattr(parser, "_option_string_actions")
    provided: Set[str] = set()
    for tok in argv:
        if not isinstance(tok, str) or not tok.startswith("-"):
            continue
        key = tok.split("=", 1)[0]  # normalize '--opt=val' -> '--opt'
        action = actions.get(key)
        if action is not None:
            provided.add(action.dest)
    return provided

def load_and_merge_config(parser: argparse.ArgumentParser) -> Tuple[argparse.Namespace, Path, Dict[str, Any]]:
    """
    Merge precedence: code defaults < config file < CLI (only flags the user passed).
    Returns (cfg Namespace, run_dir Path, effective dict).
    """
    # 1) discover --config early
    tmp, _ = parser.parse_known_args()
    cfg_from_file = load_config_file(getattr(tmp, "config", None))

    # 2) baseline from code defaults
    defaults = vars(parser.parse_args([]))

    # 3) apply file over defaults
    merged = dict(defaults)
    merged.update(cfg_from_file)

    # 4) full parse (validate types/choices) and take only CLI-provided keys
    cli_ns = parser.parse_args()
    cli_vals = vars(cli_ns)
    provided = _discover_cli_provided_dests(parser, sys.argv[1:])
    for k in provided:
        merged[k] = cli_vals[k]

    # 5) finalize
    cfg = argparse.Namespace(**merged)
    run_dir = make_run_dir(getattr(cfg, "output_dir"), getattr(cfg, "run_name", "auto"))
    return cfg, run_dir, merged