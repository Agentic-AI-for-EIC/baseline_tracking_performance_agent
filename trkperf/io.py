"""Generic ROOT/EDM4eic file access.

This module knows how to open a file (local path or root:// URL) and turn
one or more jagged EDM4eic branches into a flat pandas DataFrame. It contains
no physics whatsoever - every other module in trkperf builds on top of
`read_flat_multi`, which is the one place jagged-array flattening happens.

Why a flat DataFrame instead of working with awkward arrays throughout?
Every "which truth particle does this reconstructed track belong to"
question in this project is fundamentally a table join (event, position ->
event, position), and pandas' merge/groupby is the most widely understood
tool for that in the physics community - readability for future maintainers
mattered more here than the last bit of performance.
"""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import pandas as pd
import uproot

#: Column names that identify a row uniquely across every flat table this
#: package produces: which file it came from, which event within that file,
#: and which position within that event's collection. Every join in trkperf
#: is performed on some subset of these columns.
FILE_EVENT_IDX = ("file_id", "event", "idx")

#: Process-wide default cache directory for successfully read per-file
#: tables, set via :func:`set_cache_dir`. Kept module-level (rather than
#: threaded through every compute_* signature) so the CLI can enable caching
#: for the grid metrics without touching any of the physics modules. A cache
#: makes a relaunched run over already-processed files effectively free, which
#: is what lets a multi-hour +background analysis accumulate progress across
#: sessions even if the container/process dies mid-run (see PLAN.md).
_CACHE_DIR: str | None = None


def set_cache_dir(path: str | None) -> None:
    """Enable (or disable) caching of per-file flat tables for the rest of
    the process. ``None`` or an empty string disables it again."""
    global _CACHE_DIR
    if path:
        os.makedirs(path, exist_ok=True)
    _CACHE_DIR = path or None


def _cache_key(tree_name: str, columns: dict[str, str], path: str) -> str:
    """Stable stash key for one (file, tree, branch-set) read.

    The raw table for a given URL + columns is identical no matter which
    metric asks for it, so the key is content-addressed on exactly those
    three things - two metrics that read the same branches of the same file
    share one cached copy.
    """
    import hashlib

    material = "\x00".join([path, tree_name, repr(sorted(columns.items()))])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def open_tree(path_or_url: str, tree_name: str = "events"):
    """Open a single ROOT file and return one of its trees.

    Parameters
    ----------
    path_or_url:
        Local filesystem path, or an xrootd root://... URL. uproot handles
        both transparently.
    tree_name:
        Name of the TTree to return (default "events", the EDM4eic
        convention; "runs"/"podio_metadata" also exist but are not used by
        this package).

    Returns
    -------
    An uproot TTree-like object supporting `.arrays(...)`.
    """
    f = uproot.open(path_or_url)
    return f[tree_name]


def read_flat(tree, columns: dict[str, str], file_id: int = 0) -> pd.DataFrame:
    """Read a set of same-length jagged branches from one tree into a flat table.

    All `columns` must come from the *same* EDM4eic collection (or at least
    be aligned event-by-event and position-by-position), e.g. every member
    branch of ``MCParticles`` or every member branch of
    ``CentralCKFTrackAssociations``. Mixing branches from different
    collections here would silently misalign rows.

    Parameters
    ----------
    tree:
        Return value of :func:`open_tree`.
    columns:
        Mapping of ``{output_column_name: full_branch_path}``, e.g.
        ``{"pdg": "MCParticles/MCParticles.PDG"}``. Branch paths are the
        full podio "split" names as reported by ``tree.typenames(recursive=True)``.
    file_id:
        Integer tag stamped onto every row, so results from many files can be
        concatenated later without event-number collisions (event 0 in file 3
        is not event 0 in file 7).

    Returns
    -------
    pandas.DataFrame with columns ``file_id``, ``event``, ``idx`` (the
    position of this entry within its event's collection - this is exactly
    the value referenced by any ``_<Collection>_<relation>.index`` branch
    that points at this collection) plus one column per entry of `columns`.
    """
    branch_paths = list(columns.values())
    arrays = tree.arrays(branch_paths, library="ak")

    # Every branch of one collection has the same per-event length, so any of
    # them can be used to derive the event/idx bookkeeping columns.
    reference = arrays[branch_paths[0]]
    counts = ak.to_numpy(ak.num(reference, axis=1)).astype(np.int64)
    n_events = len(reference)

    event = np.repeat(np.arange(n_events, dtype=np.int64), counts)
    idx = ak.flatten(ak.local_index(reference, axis=1)).to_numpy().astype(np.int64)

    data: dict[str, np.ndarray] = {
        "file_id": np.full(idx.shape, file_id, dtype=np.int64),
        "event": event,
        "idx": idx,
    }
    for name, path in columns.items():
        data[name] = ak.flatten(arrays[path]).to_numpy()

    return pd.DataFrame(data)


def read_flat_multi(
    file_paths: list[str],
    columns: dict[str, str],
    tree_name: str = "events",
    *,
    max_failures: int = 0,
    cache_dir: str | None = None,
    per_file_filter: callable | None = None,
    shared_failures: dict | None = None,
) -> pd.DataFrame:
    """Read the same flat table from many files and concatenate the result.

    This is the function every higher-level reader (truth.py, reco.py,
    matching.py) actually calls. Files are processed one at a time so peak
    memory stays proportional to one file's worth of the requested branches,
    not the whole sample.

    ``per_file_filter`` (optional callable ``df -> df``) is applied to each
    file's table *after* the cache stash is taken and just before the row is
    accumulated into `frames`. It lets a caller shrink the in-memory
    accumulation (e.g. keep only ``generator_status == 1`` primaries from
    the huge +background MCParticles tables instead of hoarding every
    beam-background particle) without ever contaminating the cache: the
    stash deliberately holds the RAW table, so a different run that does not
    apply the filter still gets the complete data from the same cache entry.
    The filtered DataFrame is re-stamped (only) on the cache-load path, so
    `file_id` always reflects this run's list position.

    Parameters
    ----------
    max_failures:
        Number of individual file I/O failures to tolerate before raising.
        0 (default) means any failure is fatal — preserving the original
        strict behaviour.  A positive value allows that many files to be
        skipped (with warnings printed to stderr) while processing continues;
        set this for flaky xrootd endpoints where intermittent network errors
        are expected.  The total number of skipped files is reported in the
        warning banner so the user knows whether the result is trustworthy.
    shared_failures:
        Optional caller-owned dict for sharing one failure budget across the
        several reads of a metric run (truth + hit collections + reco +
        associations). When given, failed paths accumulate in
        ``shared_failures["skipped"]``: the budget applies to their shared
        total (not per call), and a path that already failed is skipped
        immediately by later reads without retrying or consuming more budget.
        This keeps every joined table covering the same file subset - without
        it, each read would tolerate its own failures and the truth-only
        denominators would silently cover files whose reco data is missing.
        Each metric run must pass a fresh dict.
    cache_dir:
        If set, per-file results are stashed here (pickled DataFrame, keyed
        on URL + branches) the first time they are read successfully, and a
        subsequent call for the same file re-reads the stash instead of the
        network. ``file_id`` is re-stamped on every load so the cached table
        is correct even if a different run passes the same file at a
        different list position. Cache I/O failures are logged and ignored -
        they never abort a run.

    Raises
    ------
    Exception
        Whatever uproot/awkward raises, with the offending file path
        prepended - per AGENTS.md, trkperf never silently swallows an error.
    RuntimeError
        More than ``max_failures`` files failed to read.    """
    import sys
    import time

    cache_dir = cache_dir if cache_dir is not None else _CACHE_DIR
    frames = []
    skipped: list[str] = []
    if shared_failures is not None:
        skipped = shared_failures.setdefault("skipped", [])
    for file_id, path in enumerate(file_paths):
        if path in skipped:
            # Failed earlier in this run: skip immediately so every table
            # covers the same files, without retrying or spending budget.
            continue
        cache_path: str | None = None
        if cache_dir:
            cache_path = os.path.join(
                cache_dir, _cache_key(tree_name, columns, path) + ".pkl"
            )
            if os.path.exists(cache_path):
                try:
                    df = pd.read_pickle(cache_path)
                    # The stash was written under some other run's list
                    # position; file_id is positional, so re-stamp it to this
                    # run's position before use.
                    df["file_id"] = file_id
                    df = df.reset_index(drop=True)
                    if per_file_filter is not None:
                        df = per_file_filter(df)
                    frames.append(df)
                    continue
                except Exception as exc:  # noqa: BLE001 - corrupt/foreign cache
                    print(
                        f"[io] WARNING: ignoring unusable cache entry {cache_path}: {exc}",
                        file=sys.stderr,
                    )

        last_exc: Exception | None = None
        for attempt in range(2):  # try once, retry once per AGENTS.md
            try:
                tree = open_tree(path, tree_name=tree_name)
                df = read_flat(tree, columns, file_id=file_id)
                if cache_path is not None:
                    try:
                        _stash_pickle(df, cache_path)
                    except Exception as exc:  # noqa: BLE001 - cache is best-effort
                        print(
                            f"[io] WARNING: could not stash cache entry {cache_path}: {exc}",
                            file=sys.stderr,
                        )
                if per_file_filter is not None:
                    df = per_file_filter(df)
                frames.append(df)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == 0:
                    time.sleep(1)  # brief pause before retry
        if last_exc is not None:
            skipped.append(path)
            print(
                f"[io] WARNING: skipping file ({len(skipped)} failed, "
                f"max allowed = {max_failures}): {path}\n  {last_exc}",
                file=sys.stderr,
            )
            if len(skipped) > max_failures:
                raise RuntimeError(
                    f"trkperf.io.read_flat_multi: {len(skipped)} files failed "
                    f"(max allowed={max_failures}). First failure:"
                ) from last_exc

    if skipped:
        print(
            f"[io] WARNING: {len(skipped)}/{len(file_paths)} files skipped "
            f"due to I/O errors — results may have reduced statistics.",
            file=sys.stderr,
        )

    if not frames:
        return pd.DataFrame(columns=list(FILE_EVENT_IDX) + list(columns.keys()))

    return pd.concat(frames, ignore_index=True)


def file_id_to_path_map(file_paths: list[str]) -> dict[int, str]:
    """Return the {file_id: path} mapping used by read_flat_multi, for provenance."""
    return dict(enumerate(file_paths))


# ---------------------------------------------------------------------------
# Nested (per-object vector / relation) access.
#
# ``read_flat`` covers the common case in this project - several members of ONE
# collection, all jagged with the same per-event length, flattened to one row
# per object. EDM4eic also stores two shapes that that model cannot express:
#
#   * inline vectors sized per *object* (``Cluster.shapeParameters``,
#     ``CherenkovParticleID.hypotheses``), held as a flat ``std::vector<T>`` per
#     event plus ``<field>_begin`` / ``<field>_end`` offset columns on the object
#     table, and
#   * ToOne/ToMany relations (``_EcalEndcapPClusters_hits``), held the same way
#     (flat list of ``podio::ObjectID`` + begin/end on the object table).
#
# The helpers below unpack those two shapes generically, so consumers (e.g. the
# PID feature builder) never re-implement podio's offset arithmetic. They are
# deliberately physics-free, exactly like :func:`read_flat`.
# ---------------------------------------------------------------------------


def read_nested(
    path_or_url: str,
    columns: dict[str, str],
    tree_name: str = "events",
    *,
    entry_start: int | None = None,
    entry_stop: int | None = None,
):
    """Read arbitrary branches from ONE file as (jagged) awkward arrays.

    Unlike :func:`read_flat` this performs no flattening and imposes no
    alignment requirement between the requested branches - it hands back the
    raw per-event structure so the caller can slice it with
    :func:`expand_by_offsets`. Use it for per-object vector members and
    relation lists; use :func:`read_flat`/``read_flat_multi`` for ordinary
    collection members.

    Parameters
    ----------
    path_or_url:
        Local path or ``root://`` URL (see :func:`open_tree`).
    columns:
        ``{output_name: full_branch_path}`` - same convention as
        :func:`read_flat`, e.g.
        ``{"shape": "_EcalEndcapPClusters_shapeParameters",
          "shape_begin": "EcalEndcapPClusters.shapeParameters_begin",
          "shape_end":   "EcalEndcapPClusters.shapeParameters_end"}``.
    entry_start, entry_stop:
        Optional Python-style entry slice (useful for smoke tests over a few
        events instead of the whole file).

    Returns
    -------
    ``dict`` mapping each key of `columns` to its awkward array (one entry per
    event). Branches that uproot delivers as records keep their record layout.
    """
    tree = open_tree(path_or_url, tree_name=tree_name)
    arrays = tree.arrays(
        list(columns.values()), library="ak", entry_start=entry_start, entry_stop=entry_stop
    )
    out = {}
    for name, path in columns.items():
        out[name] = arrays[path]
    return out


def event_count(path_or_url: str, tree_name: str = "events") -> int:
    """Number of entries in `tree_name`, metadata-only (no branch decompression)."""
    return int(open_tree(path_or_url, tree_name=tree_name).num_entries)


def expand_by_offsets(flat, begin, end):
    """Gather a per-event flat vector into per-object lists using begin/end.

    Parameters
    ----------
    flat:
        Jagged array of depth ``event * value`` (e.g.
        ``_EcalEndcapPClusters_shapeParameters``).
    begin, end:
        Depth ``event * object`` offset columns from the object table (e.g.
        ``EcalEndcapPClusters.shapeParameters_{begin,end}``).

    Returns
    -------
    Jagged array of depth ``event * object * value``. Empty ranges stay empty.

    Notes
    -----
    This is the generic form of "slice the inline vector / relation list that
    podio flattened across the objects of one event". Pure array arithmetic, no
    physics, no assumptions about what the values mean.
    """
    out = []
    flat_list = flat.tolist()
    begin_list = begin.tolist()
    end_list = end.tolist()
    for ev_vals, ev_b, ev_e in zip(flat_list, begin_list, end_list):
        ev_vals = ev_vals if ev_vals is not None else []
        out.append([ev_vals[b:e] for b, e in zip(ev_b, ev_e)])
    return ak.Array(out)


def _stash_pickle(df: pd.DataFrame, cache_path: str) -> None:
    """Atomically write ``df`` to ``cache_path``.

    Written to a temp file in the same directory then ``os.replace``d so a
    concurrent reader never sees a half-written pickle.
    """
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        df.to_pickle(tmp_path)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
