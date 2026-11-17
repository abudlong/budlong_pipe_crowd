"""Step timing for the crowded-field tasks.

Every timed step is stored in the task metadata (which pipetask writes to the
``<label>_metadata`` dataset) as one entry in each of these parallel arrays:

``stepLabel``
    Step name.
``stepKind``
    ``setup`` (task construction), ``io`` (butler read/write), ``compute``
    (leaf work steps) or ``wall`` (totals that contain other steps). Summing
    every non-``wall`` row gives the time accounted for.
``stepOuter``
    Detection round (1-based), or -1 outside the detection loop.
``stepInner``
    Inner index (unused at present), or -1.
``stepSeconds``
    Wall-clock duration [s].
``stepCpuSeconds``
    CPU time of this process during the step [s].
``stepStartSeconds``
    Start time relative to the timer's creation (task construction) [s].
``stepNSources``
    Number of sources the step worked on, or -1 if not meaningful.

The columns ``label``, ``seconds``, ``outer``, ``inner`` and ``kind`` match
the records of ``lsst.meas.crowd.fitLib.StepTimer`` so that runs of the two
packages can be put in one table. `timingTable` turns the metadata back into
a table.
"""

__all__ = ["StepTimer", "NullTimer", "timingTable", "summarize", "main", "TIMING_COLUMNS"]

import time
from contextlib import contextmanager, nullcontext

# metadata key -> column name in `timingTable`
TIMING_COLUMNS = {
    "stepLabel": "label",
    "stepKind": "kind",
    "stepOuter": "outer",
    "stepInner": "inner",
    "stepSeconds": "seconds",
    "stepCpuSeconds": "cpuSeconds",
    "stepStartSeconds": "startSeconds",
    "stepNSources": "nSources",
}


class StepTimer:
    """Record step timings into a task's metadata.

    Parameters
    ----------
    metadata : `lsst.pipe.base.TaskMetadata`
        Metadata to append the records to.
    log : `logging.Logger`, optional
        If given, each step is logged at VERBOSE level.
    """

    def __init__(self, metadata, log=None):
        self.metadata = metadata
        self.log = log
        self.t0 = time.perf_counter()

    @contextmanager
    def step(self, label, outer=-1, inner=-1, kind="compute", nSources=-1):
        """Time a block of code.

        Parameters
        ----------
        label : `str`
            Step name.
        outer : `int`, optional
            Detection round, or -1.
        inner : `int`, optional
            Inner index, or -1.
        kind : `str`, optional
            ``setup``, ``io``, ``compute`` or ``wall``.
        nSources : `int`, optional
            Number of sources the step works on, or -1.

        Yields
        ------
        rec : `dict`
            The record; ``nSources`` may be set inside the block (e.g. when
            the count is only known at the end).
        """
        rec = dict(nSources=nSources)
        start = time.perf_counter()
        cpuStart = time.process_time()
        try:
            yield rec
        finally:
            seconds = time.perf_counter() - start
            cpuSeconds = time.process_time() - cpuStart
            md = self.metadata
            md.add("stepLabel", str(label))
            md.add("stepKind", str(kind))
            md.add("stepOuter", int(outer))
            md.add("stepInner", int(inner))
            md.add("stepSeconds", float(seconds))
            md.add("stepCpuSeconds", float(cpuSeconds))
            md.add("stepStartSeconds", float(start - self.t0))
            md.add("stepNSources", int(rec["nSources"]))
            if self.log is not None:
                where = f" (round {outer})" if outer >= 0 else ""
                self.log.verbose("Timing: %s%s took %.3f s wall, %.3f s CPU",
                                 label, where, seconds, cpuSeconds)


class NullTimer:
    """Stand-in timer for a subtask used outside `CrowdedFieldTask`."""

    def step(self, *args, **kwargs):
        return nullcontext(dict(nSources=-1))


def _findStepMetadata(metadata):
    """Yield (task name, metadata) for every entry holding step records."""
    if "stepLabel" in metadata.keys():
        yield "", metadata
    # Full metadata has one top-level entry per task, e.g. "crowdedField",
    # "crowdedField:centroid", plus "quantum" from the executor.
    for name in metadata.keys():
        try:
            sub = metadata[name]
        except Exception:
            continue
        if hasattr(sub, "keys") and "stepLabel" in sub.keys():
            yield name, sub


def timingTable(metadata, **extraColumns):
    """Convert the step records in a task metadata object into a table.

    Also adds rows of kind ``quantum`` for the phases pipetask's executor
    records around the task (prepare, construct task, execute, total), which
    include framework overhead outside `CrowdedFieldTask.runQuantum`, and a
    ``peakRssMB`` column with the process's peak memory at the end of the
    quantum (-1 if not recorded).

    Parameters
    ----------
    metadata : `lsst.pipe.base.TaskMetadata`
        Full task metadata, e.g. ``butler.get("crowdedField_metadata", ...)``.
    **extraColumns
        Constant columns to add (e.g. ``visit=..., detector=...``).

    Returns
    -------
    table : `astropy.table.Table`
    """
    from astropy.table import Table

    rows = []
    for task, md in _findStepMetadata(metadata):
        arrays = {col: list(md.getArray(key)) for key, col in TIMING_COLUMNS.items()}
        for i in range(len(arrays["label"])):
            row = {col: arrays[col][i] for col in TIMING_COLUMNS.values()}
            row["task"] = task
            rows.append(row)

    quantumRows, peakRssMB = _quantumRows(metadata)
    rows.extend(quantumRows)

    columns = ["task"] + list(TIMING_COLUMNS.values())
    dtypes = [str, str, str, int, int, float, float, float, int]
    casts = dict(zip(columns, dtypes))
    table = Table([[casts[c](r[c]) for r in rows] for c in columns], names=columns,
                  dtype=dtypes)
    table["peakRssMB"] = [peakRssMB] * len(table)
    for name, value in extraColumns.items():
        table[name] = [value] * len(table)
    return table


# (label, start checkpoint, end checkpoint) recorded by pipetask's
# SingleQuantumExecutor in the "quantum" metadata entry.
_QUANTUM_PHASES = [
    ("pipetask: prepare quantum", "prep", "init"),
    ("pipetask: construct task", "init", "start"),
    ("pipetask: execute (runQuantum)", "start", "end"),
    ("pipetask: quantum total", "prep", "end"),
]


def _quantumRows(metadata):
    """Whole-quantum timing recorded by the pipetask executor, if present.

    Returns
    -------
    rows : `list` [`dict`]
        One row per phase in ``_QUANTUM_PHASES`` (kind ``quantum``).
    peakRssMB : `float`
        Peak resident memory of the process at the end of the quantum [MB],
        or -1 if not recorded.
    """
    from datetime import datetime
    try:
        q = metadata["quantum"]
    except Exception:
        return [], -1.0

    def get(key):
        try:
            return q[key]
        except Exception:
            return None

    rows = []
    for label, a, b in _QUANTUM_PHASES:
        ta, tb = get(f"{a}Utc"), get(f"{b}Utc")
        if ta is None or tb is None:
            continue
        try:
            seconds = (datetime.fromisoformat(str(tb)) - datetime.fromisoformat(str(ta))).total_seconds()
        except ValueError:
            continue
        ca, cb = get(f"{a}CpuTime"), get(f"{b}CpuTime")
        cpu = float(cb - ca) if ca is not None and cb is not None else -1.0
        rows.append(dict(task="quantum", label=label, kind="quantum", outer=-1, inner=-1,
                         seconds=seconds, cpuSeconds=cpu, startSeconds=-1.0, nSources=-1))
    rss = get("endMaxResidentSetSize")
    return rows, (float(rss) / 1e6 if rss is not None else -1.0)


def summarize(table, file=None):
    """Print total time per step, in the style of meas_crowd_v2's ``printTiming``.

    Parameters
    ----------
    table : `astropy.table.Table`
        Output of `timingTable` (one or more quanta).
    file : file-like, optional
        Where to print (default stdout).
    """
    from collections import defaultdict
    tot, cpu, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    kinds = {}
    for row in table:
        tot[row["label"]] += row["seconds"]
        cpu[row["label"]] += row["cpuSeconds"]
        cnt[row["label"]] += 1
        kinds[row["label"]] = row["kind"]
    print(f"\n{'step':42s} {'kind':8s} {'calls':>6s} {'wall [s]':>10s} {'CPU [s]':>10s}", file=file)
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"{k:42s} {kinds[k]:8s} {cnt[k]:6d} {v:10.2f} {cpu[k]:10.2f}", file=file)
    counted = sum(v for k, v in tot.items() if kinds[k] in ("setup", "io", "compute"))
    total = tot.get("runQuantum (total)", 0.0) + tot.get("task setup (schema + subtasks)", 0.0)
    print(f"{'(sum of setup + io + compute)':42s} {'':8s} {'':6s} {counted:10.2f}", file=file)
    if total:
        print(f"{'(in runQuantum/setup but not in a step)':42s} {'':8s} {'':6s} "
              f"{max(total - counted, 0.0):10.2f}", file=file)
    print("'wall' and 'quantum' rows contain other steps and are not added to the sum.", file=file)


def main(argv=None):
    """Command line: export the step timings of crowdedField runs to CSV."""
    import argparse
    from astropy.table import vstack
    from lsst.daf.butler import Butler

    parser = argparse.ArgumentParser(
        description="Export pipe_crowd step timings from a butler repo to a table.")
    parser.add_argument("repo", help="Butler repository.")
    parser.add_argument("collections", nargs="+", help="Collection(s) holding the run outputs.")
    parser.add_argument("--dataset", default="crowdedField_metadata",
                        help="Metadata dataset type: <pipeline task label>_metadata "
                             "(default: %(default)s).")
    parser.add_argument("--where", default="", help="Butler data ID query, e.g. 'visit = 123'.")
    parser.add_argument("-o", "--output", default="pipe_crowd_timing.csv",
                        help="Output file; format from the extension (.csv, .ecsv, .fits). "
                             "Default: %(default)s.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the summary.")
    args = parser.parse_args(argv)

    butler = Butler(args.repo, collections=args.collections)
    refs = list(butler.registry.queryDatasets(args.dataset, collections=args.collections,
                                              where=args.where, findFirst=True))
    if not refs:
        parser.error(f"No {args.dataset} datasets found in {args.collections}.")

    tables = []
    for ref in refs:
        dataId = {}
        for k in ("visit", "detector"):
            try:
                dataId[k] = ref.dataId[k]
            except (KeyError, LookupError):
                pass
        tables.append(timingTable(butler.get(ref), run=ref.run, **dataId))
    table = vstack(tables)
    table.write(args.output, overwrite=True)
    if not args.quiet:
        summarize(table)
    print(f"\n{len(refs)} quanta, {len(table)} rows -> {args.output}")
