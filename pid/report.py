"""Output writing for PID results: reuses :mod:`trkperf.report` verbatim.

The tracking metrics already established the project's output contract - JSON
with metadata, a markdown table, and a ROOT TNtuple for manual follow-up, all
under ``output/`` with species encoded as integer codes. PID results go through
the same writers so a human can treat ``pid-*`` and ``resolution-*`` files
identically; this module only adds the PID provenance block (schema version,
campaign, matching threshold, gate results, library versions).
"""

from __future__ import annotations

import os

import pandas as pd

from trkperf import report as tk_report

from . import config, dataset, features, schema


def stamp(**extra) -> dict:
    """Provenance metadata attached to every PID output."""
    env = schema.assert_ml_env(strict=False)
    base = {
        "pipeline": "pid",
        "schema_version": features.SCHEMA_VERSION,
        "campaigns": dict(config.CAMPAIGN_BY_DATASET_TAG),
        "match_weight_threshold": config.MATCH_WEIGHT_THRESHOLD,
        "min_track_p": config.MIN_TRACK_P,
        "delta_r_max": config.DELTA_R_MAX,
        "min_entries_per_bin": config.MIN_ENTRIES_PER_BIN,
        "fake_rate_targets": list(config.FAKE_RATE_TARGETS),
        "excluded_columns": list(config.EXCLUDE_COLUMNS_DEFAULT),
        "n_jobs": config.N_JOBS,
        "random_seed": config.RANDOM_SEED,
        "libraries": {k: v for k, v in env.items() if k != "numpy_path"},
        "numpy_path": env.get("numpy_path", ""),
    }
    base.update({k: v for k, v in extra.items() if _jsonable(v)})
    # AGENTS.md requires the campaign to be stated with any number; resolve it from
    # the dataset tag so a caller cannot forget, and it stays correct if the tag is
    # ever remapped.
    if "dataset_tag" in base and "campaign" not in base:
        try:
            base["campaign"] = schema.campaign_of(base["dataset_tag"])
        except ValueError:
            pass
    return base


def _jsonable(value) -> bool:
    return isinstance(value, (str, int, float, bool, list, tuple, dict, type(None)))


def encode_for_root(df: pd.DataFrame):
    """Return ``(numeric_df, legend)`` with every text column factorised to codes.

    :func:`trkperf.report.to_root` writes a TNtuple, i.e. floats only. The
    tracking metrics get away with that because their text columns are species
    names with a known code order; PID tables carry free text (``quantity``,
    ``signal``, ``bin`` intervals, ``pred_class``). Each such column becomes
    ``<name>_code`` plus a legend entry in the file's metadata, so the ROOT file
    stays queryable (``Draw("value:n_signal","quantity_code==5")``) without
    inventing a second writer.
    """
    out = df.copy()
    legend: dict[str, dict[int, str]] = {}
    for c in list(out.columns):
        series = out[c]
        if not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
            vals = [v for v in series.dropna().unique().tolist()]
            if vals and all(v in (True, False) for v in vals):
                # Nullable-boolean column (bool + NaN degrades to object):
                # write 0/1/NaN, not factorised "True"/"False" codes.
                out[c] = series.map(
                    lambda v: float(v) if pd.notna(v) else float("nan")).astype(float)
                continue
            cats = sorted({str(v) for v in vals})
            mapping = {name: i for i, name in enumerate(cats)}
            out[c + "_code"] = series.map(lambda v: float(mapping[str(v)]) if pd.notna(v) else -1.0)
            out = out.drop(columns=[c])
            legend[c] = {v: k for k, v in mapping.items()}
        elif series.dtype == bool:
            out[c] = series.astype(float)
        elif pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            out[c] = pd.to_numeric(series, errors="coerce").astype(float)
        else:
            out = out.drop(columns=[c])
    # TNtuple names must be valid C identifiers-ish; trkperf handles dots by
    # dropping, so sanitise here rather than losing the column silently.
    # Dots collide after truncation, so replace them too, and de-duplicate.
    cleaned, seen = [], {}
    for c in out.columns:
        base = (c.replace("(", "lo").replace("]", "hi").replace(",", "_")
                 .replace(" ", "").replace("/", "_").replace(".", "_")[:28])
        n = seen.get(base, 0)
        seen[base] = n + 1
        cleaned.append(base if n == 0 else f"{base}_{n + 1}"[:28])
    out.columns = cleaned
    return out, legend


def write(df: pd.DataFrame, stem: str, *, tree_name: str, meta: dict | None = None,
          markdown: bool = True) -> str:
    """Write ``<stem>.{json,md,root}`` for a PID result table."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    meta = stamp(**(meta or {}))
    tk_report.to_json(df, stem + ".json", meta=meta)
    if markdown:
        tk_report.to_markdown_table(df, stem + ".md")
    try:
        numeric, legend = encode_for_root(df)
        meta_root = dict(meta)
        for column, mapping in legend.items():
            meta_root[f"legend_{column}"] = ";".join(f"{v}:{k}" for k, v in sorted(mapping.items()))
        tk_report.to_root(numeric, stem + ".root", tree_name=tree_name.replace("-", "_"),
                          meta=meta_root)
        root_ok = True
    except Exception as exc:  # noqa: BLE001 - a TNtuple is a convenience, not a gate
        print(f"[report] WARNING: ROOT TNtuple skipped for {stem}: "
              f"{exc.__class__.__name__}: {exc}")
        root_ok = False
    print(f"[report] wrote {stem}.json"
          + (f", {stem}.md" if markdown else "")
          + (f", {stem}.root" if root_ok else "")
          + f" ({len(df)} rows)")
    return stem


def read(stem_or_json: str) -> tuple[pd.DataFrame, dict]:
    """Read back a PID result written by :func:`write`."""
    path = stem_or_json if stem_or_json.endswith(".json") else stem_or_json + ".json"
    return tk_report.read_json(path)


def load_features_table(path: str) -> pd.DataFrame:
    return dataset.load_table(path)
