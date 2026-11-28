#!/usr/bin/env python3
"""End-to-end unit tests for the popsift-compareSiftFiles executable.

Usage:
    test_compare_sift_files.py --tool <executable> <case>

Each test writes two deterministic text feature files and invokes the
executable with the requested options. A feature row contains x, y, sigma,
and orientation in radians, followed by exactly 128 descriptor values.

The harness verifies process exit status, pair-file row counts and values,
and selected output text. Each check is printed as PASS or FAIL. A successful
run exits with status 0; a run with any failed check exits with status 1.

Available cases:
    agreement: Verify the three co-location agreement levels and tolerance
        overrides.
    identical: Verify that identical features compare at zero distance.
    shifted: Verify that a horizontal shift affects position distance and dx.
    missing: Verify that unmatched features are reported in the A-only and
        B-only files.
    empty: Verify empty-file handling with and without --allow-empty.
    orientation-wrap: Verify orientation differences across the 0/360-degree
        boundary.
    same-position: Verify that multiple orientations at one position are
        paired by orientation rather than file order.
    max-avg-dist: Verify average descriptor distance and descriptor-bin sums,
        as well as --max-avg-dist.
"""

import argparse
import csv
import math
import os
import random
import subprocess  # nosec B404
import sys
from collections.abc import Callable

DESC_SIZE: int = 128


def feature(x: float, y: float, sigma: float, ori_deg: float, seed: int) -> list[float]:
    """Return a feature with a reproducible random descriptor.

    The descriptor is generated from an isolated random number generator, so
    the same arguments always produce the same 128 values. Generated values
    are integers in the inclusive range 0 through 255, converted to floats.

    Args:
        x: Horizontal keypoint position.
        y: Vertical keypoint position.
        sigma: Keypoint scale.
        ori_deg: Keypoint orientation in degrees.
        seed: Seed for the reproducible descriptor.

    Returns:
        The keypoint values followed by 128 descriptor values.
    """
    rng = random.Random(seed)  # nosec B311
    return [x, y, sigma, math.radians(ori_deg)] + [
        float(rng.randrange(0, 256)) for _ in range(DESC_SIZE)
    ]


# Five features that are far apart, with different descriptors.
BASE: list[list[float]] = [
    feature(20.0 + 30 * i, 15.0 + 7 * i, 2.0 + 0.5 * i, 40.0 * i, i) for i in range(5)
]


class Case:
    """One test case: its folder, the tool, and the count of failed checks.

    Creating a case creates a directory for generated feature and comparison
    files. Re-running a case clears previous pair CSV files before each tool
    invocation.

    Attributes:
        tool: Path to the compareSiftFiles executable.
        dir: Directory containing the test case files.
        failures: Number of failed checks.
    """

    def __init__(self, tool: str, work_dir: str, name: str) -> None:
        """Create the folder of the case below work_dir.

        The directory is created when it does not exist. Existing files from
        prior runs are retained except that pair CSV files are removed by
        :meth:`run` before each executable invocation.

        Args:
            tool: Path to the compareSiftFiles executable.
            work_dir: Directory in which to create the case directory.
            name: Name of the test case.
        """
        self.tool: str = tool
        self.dir: str = os.path.join(work_dir, name)
        os.makedirs(self.dir, exist_ok=True)
        self.failures: int = 0

    def write(self, name: str, features: list[list[float]]) -> str:
        """Write features into a file of the case.

        Rows are written with six decimal places and a single ASCII space
        separating values. This method creates no directory; the case directory
        must already exist.

        Args:
            name: Name of the feature file.
            features: Keypoints and descriptors to write.

        Returns:
            Path to the written feature file.
        """
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(" ".join(f"{v:.6f}" for v in feat) + "\n" for feat in features)
        return path

    def run(
        self, a: list[list[float]], b: list[list[float]], *options: str
    ) -> tuple[int, str]:
        """Run the tool on two feature sets.

        The A set is written to ``a.sift`` and the B set to ``b.sift``. The
        process receives ``--pair-dir`` for pair CSV output and ``-o`` for the
        comparison text file. The pair directory is cleared before invocation.
        Standard output and standard error are merged and decoded as UTF-8
        replacement text.

        Args:
            a: Features for the A input file.
            b: Features for the B input file.
            *options: Additional command-line options.

        Returns:
            A tuple containing the process exit code and combined output.

        Raises:
            FileNotFoundError: If the configured executable cannot be started.
            PermissionError: If the configured executable cannot be executed.
        """
        pair_dir = os.path.join(self.dir, "pairs")
        os.makedirs(pair_dir, exist_ok=True)
        for name in os.listdir(pair_dir):
            os.remove(os.path.join(pair_dir, name))
        cmd = [
            self.tool,
            "--pair-dir",
            pair_dir,
            "-o",
            os.path.join(self.dir, "comparison.txt"),
            *options,
            self.write("a.sift", a),
            self.write("b.sift", b),
        ]
        print("   " + " ".join(cmd))
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,  # nosec B603
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = proc.stdout.decode("utf-8", "replace")
        for line in output.splitlines():
            print("   | " + line)
        return proc.returncode, output

    def rows(self, name: str) -> list[dict[str, str]]:
        """Return the rows of a pair file.

        Missing files are interpreted as an empty result. Header names and
        cell values are read as strings; missing cells are represented by empty
        strings.

        Args:
            name: Name of the pair file.

        Returns:
            CSV rows as dictionaries, or an empty list if the file is absent.
        """
        path = os.path.join(self.dir, "pairs", name)
        if not os.path.exists(path):
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return [
                {key: value or "" for key, value in row.items()}
                for row in csv.DictReader(f)
            ]

    def check(self, ok: bool, what: str) -> None:
        """Print PASS or FAIL for one check and count failures.

        This method prints exactly one line in the form ``PASS: <what>`` or
        ``FAIL: <what>`` and increments ``failures`` only for a failing check.

        Args:
            ok: Whether the check passed.
            what: Description of the check.
        """
        print(f"{'PASS' if ok else 'FAIL'}: {what}")
        if not ok:
            self.failures += 1

    def check_close(
        self, values: list[float], expected: float, tol: float, what: str
    ) -> None:
        """Check that values exist and are within tolerance of an expected value.

        The check passes only when ``values`` contains at least one item and
        every item is within ``tol`` of ``expected`` using absolute difference.
        The reported pair count is the number of input values.

        Args:
            values: Values to check.
            expected: Expected value.
            tol: Allowed absolute tolerance.
            what: Description of the values.
        """
        bad = [v for v in values if abs(v - expected) > tol]
        self.check(
            bool(values) and not bad,
            f"{what} is {expected} (+-{tol}) for all {len(values)} "
            f"pair(s)" + (f", got {bad[:5]}" if bad else ""),
        )

    def check_counts(self, colocated: int, only_a: int, only_b: int) -> None:
        """Check the number of rows in the three pair files.

        This check reads ``colocated.csv``, ``only_a.csv``, and ``only_b.csv``,
        in that order, and passes only when their respective row counts equal
        the three arguments.

        Args:
            colocated: Expected co-located pair count.
            only_a: Expected A-only pair count.
            only_b: Expected B-only pair count.
        """
        got = (
            len(self.rows("colocated.csv")),
            len(self.rows("only_a.csv")),
            len(self.rows("only_b.csv")),
        )
        self.check(
            got == (colocated, only_a, only_b),
            f"{colocated} co-located, {only_a} only in a, {only_b} only "
            f"in b (got {got[0]}, {got[1]}, {got[2]})",
        )

    def agreement(self) -> dict[int, str]:
        """Return co-located agreement values indexed by A keypoint.

        Each ``a_index`` value in ``colocated.csv`` is converted to an integer
        and maps to the corresponding string ``agreement`` value.

        Returns:
            Agreement level for each A keypoint index.
        """
        return {int(r["a_index"]): r["agreement"] for r in self.rows("colocated.csv")}

    def column(self, name: str, file: str = "colocated.csv") -> list[float]:
        """Return one pair-file column as floats.

        A row is included once for every row in the named pair file. The
        selected cell is parsed by ``float``.

        Args:
            name: Column name.
            file: Pair-file name.

        Returns:
            Column values converted to floats.
        """
        return [float(r[name]) for r in self.rows(file)]


def case_identical(c: Case) -> None:
    """Verify behavior for two byte-equivalent feature lists.

    Setup:
        Invoke the executable once with ``BASE`` as both the A and B input.

    Assertions:
        The exit status is 0. Exactly five co-located pairs and no unmatched
        pairs are produced. Position and descriptor distances are 0 within
        tolerances of 1e-6. Descriptor nearest-neighbor indices equal the
        co-located indices. Every agreement value is ``descriptor``. The
        combined output contains ``Average best-match descriptor distance: 0``.

    Args:
        c: Test case state and helpers.
    """
    code, output = c.run(BASE, BASE)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(5, 0, 0)
    c.check_close(c.column("pos_dist"), 0.0, 1e-6, "position distance")
    c.check_close(c.column("desc_dist"), 0.0, 1e-6, "descriptor distance")
    nn = c.rows("desc_nn_a.csv")
    c.check(
        all(r["n_index"] == r["colocated_index"] for r in nn),
        "every descriptor nearest neighbour is the co-located keypoint",
    )
    c.check(
        set(c.agreement().values()) == {"descriptor"},
        "every pair agrees up to the descriptor",
    )
    c.check(
        "Average best-match descriptor distance: 0" in output,
        "average best-match descriptor distance 0 is printed",
    )


def case_shifted(c: Case) -> None:
    """Verify that a 0.5-pixel horizontal shift is reported independently.

    Setup:
        Create B by adding 0.5 to the x coordinate of every feature in ``BASE``.

    Assertions:
        The exit status is 0. Exactly five co-located pairs and no unmatched
        pairs are produced. ``pos_dist`` and ``dx`` are 0.5 within 1e-4.
        ``desc_dist`` remains 0 within 1e-6.

    Args:
        c: Test case state and helpers.
    """
    b = [[f[0] + 0.5] + f[1:] for f in BASE]
    code, _ = c.run(BASE, b)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(5, 0, 0)
    c.check_close(c.column("pos_dist"), 0.5, 1e-4, "position distance")
    c.check_close(c.column("dx"), 0.5, 1e-4, "dx (b - a)")
    c.check_close(c.column("desc_dist"), 0.0, 1e-6, "descriptor distance")


def case_missing(c: Case) -> None:
    """Verify unmatched-feature reporting in both input directions.

    Setup:
        First run ``BASE`` as A and ``BASE[:3]`` as B. Then run ``BASE[1:]``
        as A and ``BASE`` as B.

    Assertions:
        Both invocations exit with status 0. In the first invocation, exactly
        three co-located pairs and features with indices 3 and 4 in only-A are
        produced. In the second, exactly four co-located pairs and one only-B
        feature are produced.

    Args:
        c: Test case state and helpers.
    """
    code, _ = c.run(BASE, BASE[:3])
    c.check(code == 0, f"exit code 0 with features missing in b (got {code})")
    c.check_counts(3, 2, 0)
    only_a = sorted(int(r["index"]) for r in c.rows("only_a.csv"))
    c.check(only_a == [3, 4], f"only_a.csv lists features 3 and 4 (got {only_a})")
    code, _ = c.run(BASE[1:], BASE)
    c.check(code == 0, f"exit code 0 with features missing in a (got {code})")
    c.check_counts(4, 0, 1)


def case_empty(c: Case) -> None:
    """Verify empty-input behavior with and without ``--allow-empty``.

    Setup:
        Run two empty inputs, then repeat them with ``--allow-empty``.
        Separately run ``BASE`` with an empty B and an empty A with ``BASE``,
        each with ``--allow-empty``.

    Assertions:
        Two empty inputs fail without ``--allow-empty`` and succeed with it.
        One empty input fails even with ``--allow-empty``. Each failure exit
        status is nonzero and combined output contains ``FAIL``.

    Args:
        c: Test case state and helpers.
    """
    code, output = c.run([], [])
    c.check(
        code != 0 and "FAIL" in output,
        f"two empty files fail without --allow-empty (exit code {code})",
    )
    code, _ = c.run([], [], "--allow-empty")
    c.check(code == 0, f"two empty files pass with --allow-empty (got {code})")
    code, output = c.run(BASE, [], "--allow-empty")
    c.check(
        code != 0 and "FAIL" in output,
        f"an empty b fails even with --allow-empty (exit code {code})",
    )
    code, output = c.run([], BASE, "--allow-empty")
    c.check(
        code != 0 and "FAIL" in output,
        f"an empty a fails even with --allow-empty (exit code {code})",
    )


def case_orientation_wrap(c: Case) -> None:
    """Verify shortest angular differences across the 0/360-degree boundary.

    Setup:
        Run three pairs with A/B orientations of 0.5/359.5,
        359.5/0.5, and -10/350 degrees.

    Assertions:
        The exit status is 0 and all three pairs are co-located. Differences
        ``b - a`` are -1, 1, and 0 degrees, respectively, within 1e-3.
        All written A and B orientation values are in the interval [0, 360).

    Args:
        c: Test case state and helpers.
    """
    a = [
        BASE[0][:3] + [math.radians(0.5)] + BASE[0][4:],
        BASE[1][:3] + [math.radians(359.5)] + BASE[1][4:],
        BASE[2][:3] + [math.radians(-10.0)] + BASE[2][4:],
    ]
    b = [
        BASE[0][:3] + [math.radians(359.5)] + BASE[0][4:],
        BASE[1][:3] + [math.radians(0.5)] + BASE[1][4:],
        BASE[2][:3] + [math.radians(350.0)] + BASE[2][4:],
    ]
    code, _ = c.run(a, b)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(3, 0, 0)
    diffs = {
        int(r["a_index"]): float(r["ori_diff_deg"]) for r in c.rows("colocated.csv")
    }
    expected = {0: -1.0, 1: 1.0, 2: 0.0}
    c.check(
        all(abs(diffs.get(i, 999) - d) < 1e-3 for i, d in expected.items()),
        f"orientation differences (b - a) are {expected} (got {diffs})",
    )
    oris = c.column("a_ori_deg") + c.column("b_ori_deg")
    c.check(
        all(0.0 <= o < 360.0 for o in oris),
        f"orientations are written in [0, 360) degrees (got {oris})",
    )


def case_same_position(c: Case) -> None:
    """Verify pairing of multiple orientations at the same position.

    Setup:
        Create three A features at position (0, 0) with orientations 10, 100,
        and 200 degrees and distinct descriptors. Use those same three features
        in B in reverse file order.

    Assertions:
        The exit status is 0 and exactly three co-located pairs are produced.
        Orientation differences are 0 within 1e-3, and descriptor distances are
        0 within 1e-6. Pairing must be independent of the reversed B order.

    Args:
        c: Test case state and helpers.
    """
    a = [
        BASE[0][:3] + [math.radians(ori)] + feature(0, 0, 0, 0, 10 + k)[4:]
        for k, ori in enumerate((10.0, 100.0, 200.0))
    ]
    b = [a[2], a[0], a[1]]
    code, _ = c.run(a, b)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(3, 0, 0)
    c.check_close(c.column("ori_diff_deg"), 0.0, 1e-3, "orientation difference")
    c.check_close(c.column("desc_dist"), 0.0, 1e-6, "descriptor distance")


def case_max_avg_dist(c: Case) -> None:
    """Verify average descriptor distance, descriptor bins, and its limit.

    Setup:
        Create B by adding 6 to descriptor element 4 of every feature in
        ``BASE``. Invoke once without a limit, then with ``--max-avg-dist 7``,
        and finally with ``--max-avg-dist 5``.

    Assertions:
        Without a limit and at limit 7, the exit status is 0 and combined
        output reports an average best-match descriptor distance of 6.
        ``desc_bins.csv`` contains 256 entries (2 subsets by 128 elements).
        Element 0 has sum 30 in both co-located and same-orientation subsets;
        every other element has sum 0 in absolute-difference terms.
        A limit of 5 fails with a nonzero exit status and ``FAIL`` output.

    Args:
        c: Test case state and helpers.
    """
    # Each descriptor of b differs by 6 in one component: L2 distance 6.
    b = [f[:4] + [f[4] + 6.0] + f[5:] for f in BASE]
    code, output = c.run(BASE, b)
    c.check(code == 0, f"exit code 0 without a limit (got {code})")
    c.check(
        "Average best-match descriptor distance: 6" in output,
        "average best-match descriptor distance 6 is printed",
    )
    bins = {(r["subset"], int(r["element"])): r for r in c.rows("desc_bins.csv")}
    c.check(len(bins) == 256, f"desc_bins.csv has 2 x 128 rows (got {len(bins)})")
    first = [
        bins.get((s, 0), {}).get("sum_diff") for s in ("colocated", "same_orientation")
    ]
    c.check(
        first == ["30", "30"],
        f"element 0 sums to 5 x 6 = 30 in both subsets (got {first})",
    )
    rest = {float(r["sum_abs_diff"]) for (s, i), r in bins.items() if i > 0}
    c.check(rest == {0.0}, f"all other elements sum to 0 (got {sorted(rest)[:5]})")
    code, output = c.run(BASE, b, "--max-avg-dist", "7")
    c.check(code == 0, f"passes with --max-avg-dist 7 (got {code})")
    code, output = c.run(BASE, b, "--max-avg-dist", "5")
    c.check(
        code != 0 and "FAIL" in output,
        f"fails with --max-avg-dist 5 (exit code {code})",
    )


def case_agreement(c: Case) -> None:
    """Verify agreement levels and tolerance overrides.

    Setup:
        Copy ``BASE`` and modify five features:

        0. No change.
        1. Rotate by 30 degrees.
        2. Increase descriptor element 4 by 20.
        3. Increase descriptor element 4 by 10.
        4. Rotate by 1 degree and replace the descriptor.

        Run once with default tolerances, then rerun with orientation tolerance
        45 degrees and descriptor tolerance 25.

    Assertions:
        The default mapping is 0: descriptor, 1: position, 2: orientation,
        3: descriptor, 4: orientation. Exactly five co-located pairs are
        produced. Output reports four same-orientation pairs and two pairs also
        having similar descriptors. With wider tolerances, indices 1 and 2
        become ``descriptor`` and index 4 remains ``orientation``.

    Args:
        c: Test case state and helpers.
    """
    b = [list(f) for f in BASE]
    b[1][3] += math.radians(30.0)
    b[2][4] += 20.0
    b[3][4] += 10.0
    b[4] = b[4][:3] + [b[4][3] + math.radians(1.0)] + feature(0, 0, 0, 0, 99)[4:]
    code, output = c.run(BASE, b)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(5, 0, 0)
    expected = {
        0: "descriptor",
        1: "position",
        2: "orientation",
        3: "descriptor",
        4: "orientation",
    }
    got = c.agreement()
    c.check(got == expected, f"agreement levels are {expected} (got {got})")
    c.check(
        "same orientation (<= 2 deg): 4" in output
        and "also similar descriptor (<= 12): 2" in output,
        "the counts of both levels are printed",
    )
    code, _ = c.run(BASE, b, "--ori-tolerance-deg", "45", "--desc-tolerance", "25")
    got = c.agreement()
    c.check(
        got[1] == "descriptor" and got[2] == "descriptor" and got[4] == "orientation",
        f"wider tolerances raise pairs 1 and 2 to descriptor (got {got})",
    )


def case_best_at_position(c):
    """Check best_at_position, the best level among the orientations of one
    position.

    Position 0 has three orientations: the first agrees, the second has a
    different descriptor, the third is rotated by 30 degrees in b. All three
    have the best level descriptor. Position 1 has one keypoint whose
    descriptor differs, so its best level is orientation.
    """
    a = [BASE[0][:3] + [math.radians(ori)] + feature(0, 0, 0, 0, 20 + k)[4:]
         for k, ori in enumerate((10.0, 100.0, 200.0))] + [BASE[1]]
    b = [list(f) for f in a]
    b[1][4] += 20.0
    b[2][3] += math.radians(30.0)
    b[3][4] += 20.0
    code, output = c.run(a, b)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(4, 0, 0)
    rows = {int(r["a_index"]): (r["agreement"], r["best_at_position"])
            for r in c.rows("colocated.csv")}
    expected = {0: ("descriptor", "descriptor"),
                1: ("orientation", "descriptor"),
                2: ("position", "descriptor"),
                3: ("orientation", "orientation")}
    c.check(rows == expected,
            f"agreement and best_at_position are {expected} (got {rows})")
    c.check("orientation differs, but another keypoint at the same "
            "position agrees: 1" in output
            and "descriptor differs, but another keypoint at the same "
            "position agrees: 2" in output,
            "the pairs with a better pair at their position are counted")


CASES: dict[str, Callable[[Case], None]] = {
    "agreement": case_agreement,
    "best-at-position": case_best_at_position,
    "identical": case_identical,
    "shifted": case_shifted,
    "missing": case_missing,
    "empty": case_empty,
    "orientation-wrap": case_orientation_wrap,
    "same-position": case_same_position,
    "max-avg-dist": case_max_avg_dist,
}


def main() -> int:
    """Parse command-line arguments and run the selected test case.

    ``--tool`` is required and names the executable. ``--work-dir`` defaults
    to the current directory. ``case`` must be one of the keys in ``CASES``.
    The selected case may perform one or more executable invocations and record
    failures on the returned :class:`Case`.

    Returns:
        0 if all checks pass; 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tool", required=True, help="the popsift-compareSiftFiles binary"
    )
    parser.add_argument(
        "--work-dir", default=".", help="where the feature and match files go"
    )
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    c = Case(args.tool, args.work_dir, args.case)
    CASES[args.case](c)
    print(
        f"{args.case}: {'PASSED' if c.failures == 0 else 'FAILED'}"
        + (f", {c.failures} check(s) failed" if c.failures else "")
    )
    return 0 if c.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
