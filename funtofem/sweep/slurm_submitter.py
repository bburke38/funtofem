"""
funtofem.sweep.slurm_submitter — SLURM job submission for parameter sweeps.

Provides SlurmSweepSubmitter, which writes a point.json file for each design
point and submits a per-point SLURM batch job via sbatch.  Supports dry-run
mode and skip-completed logic.
"""

import csv
import json
import os
import subprocess

from .parameter_sweep import _make_key


class SlurmSweepSubmitter:
    """Submit one SLURM batch job per design point in a :class:`ParameterSweep`.

    For each design point the submitter:

    1. Writes ``point.json`` to ``<cfd_root>/<key>/point.json``.
    2. Builds a SLURM batch script that passes ``--point-file`` to the user's
       sweep script.
    3. Submits the script via ``sbatch`` (or prints it when ``dry_run=True``).
    4. Writes a submission summary text file alongside the result CSV.

    Parameters
    ----------
    sweep : ParameterSweep
        Configured :class:`ParameterSweep` instance.  Used to enumerate design
        points and to locate the result CSV and ``cfd_root``.
    slurm_config : dict
        SLURM scheduler options.  Required keys: ``account``, ``partition``,
        ``qos``, ``nodes``, ``ntasks_per_node``, ``walltime``.
    sweep_script : str
        Path to the user's Python sweep script that accepts ``--point-file``.
    preamble : str
        Shell lines inserted before the ``srun`` invocation (module loads,
        environment setup, etc.).  Defaults to an empty string.
    log_dir : str
        Directory name used for per-job stdout/stderr files.  Defaults to
        ``"sweep_logs"``.  The submitter does **not** create this directory;
        SLURM creates it when the job starts, or the user must pre-create it.
    extra_directives : list[str] or None
        Additional ``#SBATCH`` directives injected verbatim into each script
        (e.g. ``["--mail-type=END", "--mail-user=you@example.com"]``).
        Defaults to an empty list when ``None``.

    Examples
    --------
    >>> submitter = SlurmSweepSubmitter(
    ...     sweep,
    ...     slurm_config={
    ...         "account": "MY_ACCOUNT",
    ...         "partition": "general",
    ...         "qos": "standard",
    ...         "nodes": 1,
    ...         "ntasks_per_node": 128,
    ...         "walltime": "08:00:00",
    ...     },
    ...     sweep_script="sweep.py",
    ...     preamble="source $HOME/.bashrc",
    ...     log_dir="sweep_logs",
    ...     extra_directives=["--mail-type=END"],
    ... )
    >>> submitter.submit(dry_run=True)
    """

    def __init__(
        self,
        sweep,
        slurm_config: dict,
        *,
        sweep_script: str,
        preamble: str = "",
        log_dir: str = "sweep_logs",
        extra_directives: "list[str] | None" = None,
    ) -> None:
        self.sweep = sweep
        self.slurm_config = slurm_config
        self.sweep_script = sweep_script
        self.preamble = preamble
        self.log_dir = log_dir
        self.extra_directives = extra_directives if extra_directives is not None else []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        dry_run: bool = False,
        skip_completed: bool = False,
    ) -> None:
        """Submit (or preview) one SLURM job per design point.

        Parameters
        ----------
        dry_run : bool
            When ``True``, print each job script to stdout without calling
            ``sbatch``.  ``point.json`` files are still written.
        skip_completed : bool
            When ``True``, read the result CSV and skip design points that
            already have ``status = "success"``.
        """
        sweep = self.sweep
        design_points = sweep.strategy.generate(sweep.params)

        # Build set of already-completed keys
        completed_keys: "set[str]" = set()
        if skip_completed:
            csv_path = sweep.output_csv
            if os.path.isfile(csv_path):
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("status") == "success":
                            completed_keys.add(row.get("key", ""))

        # Submission results: list of (key, status, job_id_or_reason)
        submission_results: "list[tuple[str, str, str]]" = []

        for design_point in design_points:
            key = _make_key(design_point, sweep.key_fn)

            if key in completed_keys:
                print(f"[slurm] SKIP {key} (already completed)")
                submission_results.append((key, "skipped", "already completed"))
                continue

            # 1. Write point.json
            point_dir = os.path.join(sweep.cfd_root, key)
            os.makedirs(point_dir, exist_ok=True)
            point_json_path = os.path.join(point_dir, "point.json")
            with open(point_json_path, "w") as f:
                json.dump(design_point, f)

            # 2. Build the job script
            script_str = self._build_job_script(key, point_json_path)

            if dry_run:
                # 3a. Dry run — print and record without submitting
                print(f"[slurm] DRY RUN {key}")
                print("-" * 60)
                print(script_str)
                print("-" * 60)
                submission_results.append((key, "dry_run", ""))
            else:
                # 3b. Submit via sbatch
                result = subprocess.run(
                    ["sbatch"],
                    input=script_str,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    print(f"[slurm] SUBMITTED {key} → job {job_id}")
                    submission_results.append((key, "submitted", job_id))
                else:
                    reason = result.stderr.strip()
                    print(f"[slurm] sbatch FAILED for key={key}: {reason}")
                    submission_results.append((key, "failed", reason))

        # 4. Write submission summary
        self._write_summary(submission_results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_job_script(self, key: str, point_json_path: str) -> str:
        """Build the SLURM batch script string for a single design point.

        Parameters
        ----------
        key : str
            Design key for the point (used as the SLURM job name and log stem).
        point_json_path : str
            Absolute or relative path to the ``point.json`` file for this point.

        Returns
        -------
        str
            Complete SLURM batch script as a string, ready to pipe into ``sbatch``.
        """
        cfg = self.slurm_config
        account = cfg["account"]
        partition = cfg["partition"]
        qos = cfg["qos"]
        nodes = cfg["nodes"]
        ntasks_per_node = cfg["ntasks_per_node"]
        walltime = cfg["walltime"]
        log_dir = self.log_dir

        # Extra directives — one #SBATCH line per entry
        extra_lines = "\n".join(
            f"#SBATCH {directive}" for directive in self.extra_directives
        )
        extra_block = f"\n{extra_lines}" if extra_lines else ""

        # Preamble block
        preamble_block = f"\n{self.preamble}\n" if self.preamble else "\n"

        script = (
            f"#!/bin/bash\n"
            f"#SBATCH -A {account}\n"
            f"#SBATCH --job-name={key}\n"
            f"#SBATCH -p {partition}\n"
            f"#SBATCH -q {qos}\n"
            f"#SBATCH --nodes={nodes}\n"
            f"#SBATCH --ntasks-per-node={ntasks_per_node}\n"
            f"#SBATCH --time={walltime}\n"
            f"#SBATCH --output={log_dir}/{key}.out\n"
            f"#SBATCH --error={log_dir}/{key}.err"
            f"{extra_block}"
            f"{preamble_block}"
            f"srun python {self.sweep_script} --point-file {point_json_path}\n"
        )
        return script

    def _write_summary(self, results: "list[tuple[str, str, str]]") -> None:
        """Write the submission summary text file alongside the result CSV.

        The summary records each job's design key, submission status, and SLURM
        job ID (or failure reason / empty string for dry-run).

        Parameters
        ----------
        results : list[tuple[str, str, str]]
            Each element is ``(key, status, job_id_or_reason)`` where *status*
            is one of ``"submitted"``, ``"dry_run"``, ``"skipped"``,
            ``"failed"``.
        """
        summary_path = f"{self.sweep.output_csv}.submission_summary.txt"
        with open(summary_path, "w") as f:
            f.write("key\tstatus\tjob_id_or_reason\n")
            f.write("-" * 60 + "\n")
            for key, status, detail in results:
                f.write(f"{key}\t{status}\t{detail}\n")
        print(f"[slurm] Submission summary written to: {summary_path}")
