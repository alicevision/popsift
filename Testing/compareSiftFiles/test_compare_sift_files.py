#!/usr/bin/env python3
"""Unit tests of popsift-compareSiftFiles on small hand-made feature files.

    test_compare_sift_files.py --tool <popsift-compareSiftFiles> <case>

Each case writes two feature files (x y sigma orientation in radians, then
128 descriptor values per line), runs the tool on them and checks its exit
code and its match files. Every check is printed with its outcome; the exit
code is 0 if all of them passed.
"""

import argparse
import csv
import math
import os
import random
import subprocess
import sys

DESC_SIZE = 128


def feature(x, y, sigma, ori_deg, seed):
    """A feature with a reproducible random descriptor."""
    rng = random.Random(seed)
    return [x, y, sigma, math.radians(ori_deg)] + \
        [float(rng.randrange(0, 256)) for _ in range(DESC_SIZE)]


# Five features that are far apart, with different descriptors.
BASE = [feature(20.0 + 30 * i, 15.0 + 7 * i, 2.0 + 0.5 * i, 40.0 * i, i)
        for i in range(5)]


class Case:
    def __init__(self, tool, work_dir, name):
        self.tool = tool
        self.dir = os.path.join(work_dir, name)
        os.makedirs(self.dir, exist_ok=True)
        self.failures = 0

    def write(self, name, features):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for feat in features:
                f.write(" ".join(f"{v:.6f}" for v in feat) + "\n")
        return path

    def run(self, a, b, *options):
        """Runs the tool on the features a and b. Returns (exit code, output)."""
        match_dir = os.path.join(self.dir, "match")
        os.makedirs(match_dir, exist_ok=True)
        for name in os.listdir(match_dir):
            os.remove(os.path.join(match_dir, name))
        cmd = [self.tool, "--match-dir", match_dir,
               "-o", os.path.join(self.dir, "comparison.txt"),
               *options, self.write("a.sift", a), self.write("b.sift", b)]
        print("   " + " ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        output = proc.stdout.decode("utf-8", "replace")
        for line in output.splitlines():
            print("   | " + line)
        return proc.returncode, output

    def rows(self, name):
        path = os.path.join(self.dir, "match", name)
        if not os.path.exists(path):
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def check(self, ok, what):
        print(f"{'PASS' if ok else 'FAIL'}: {what}")
        if not ok:
            self.failures += 1

    def check_close(self, values, expected, tol, what):
        bad = [v for v in values if abs(v - expected) > tol]
        self.check(values and not bad,
                   f"{what} is {expected} (+-{tol}) for all {len(values)} "
                   f"pair(s)" + (f", got {bad[:5]}" if bad else ""))

    def check_counts(self, matched, only_a, only_b):
        got = (len(self.rows("matched.csv")), len(self.rows("only_a.csv")),
               len(self.rows("only_b.csv")))
        self.check(got == (matched, only_a, only_b),
                   f"{matched} matched, {only_a} only in a, {only_b} only in "
                   f"b (got {got[0]}, {got[1]}, {got[2]})")

    def column(self, name, file="matched.csv"):
        return [float(r[name]) for r in self.rows(file)]


def case_identical(c):
    code, output = c.run(BASE, BASE)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(5, 0, 0)
    c.check_close(c.column("pos_dist"), 0.0, 1e-6, "position distance")
    c.check_close(c.column("desc_dist"), 0.0, 1e-6, "descriptor distance")
    nn = c.rows("desc_nn_a.csv")
    c.check(all(r["n_index"] == r["geo_index"] for r in nn),
            "every descriptor nearest neighbour is the geometric match")
    c.check("Average best-match descriptor distance: 0" in output,
            "average best-match descriptor distance 0 is printed")


def case_shifted(c):
    b = [[f[0] + 0.5] + f[1:] for f in BASE]
    code, _ = c.run(BASE, b)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(5, 0, 0)
    c.check_close(c.column("pos_dist"), 0.5, 1e-4, "position distance")
    c.check_close(c.column("dx"), 0.5, 1e-4, "dx (b - a)")
    c.check_close(c.column("desc_dist"), 0.0, 1e-6, "descriptor distance")


def case_missing(c):
    code, _ = c.run(BASE, BASE[:3])
    c.check(code == 0, f"exit code 0 with features missing in b (got {code})")
    c.check_counts(3, 2, 0)
    only_a = sorted(int(r["index"]) for r in c.rows("only_a.csv"))
    c.check(only_a == [3, 4], f"only_a.csv lists features 3 and 4 (got {only_a})")
    code, _ = c.run(BASE[1:], BASE)
    c.check(code == 0, f"exit code 0 with features missing in a (got {code})")
    c.check_counts(4, 0, 1)


def case_empty(c):
    code, output = c.run([], [])
    c.check(code != 0 and "FAIL" in output,
            f"two empty files fail without --allow-empty (exit code {code})")
    code, _ = c.run([], [], "--allow-empty")
    c.check(code == 0, f"two empty files pass with --allow-empty (got {code})")
    code, output = c.run(BASE, [], "--allow-empty")
    c.check(code != 0 and "FAIL" in output,
            f"an empty b fails even with --allow-empty (exit code {code})")
    code, output = c.run([], BASE, "--allow-empty")
    c.check(code != 0 and "FAIL" in output,
            f"an empty a fails even with --allow-empty (exit code {code})")


def case_orientation_wrap(c):
    # 0.5 and 359.5 degrees are 1 degree apart, not 359. A negative angle
    # is the same as the angle plus 360 degrees.
    a = [BASE[0][:3] + [math.radians(0.5)] + BASE[0][4:],
         BASE[1][:3] + [math.radians(359.5)] + BASE[1][4:],
         BASE[2][:3] + [math.radians(-10.0)] + BASE[2][4:]]
    b = [BASE[0][:3] + [math.radians(359.5)] + BASE[0][4:],
         BASE[1][:3] + [math.radians(0.5)] + BASE[1][4:],
         BASE[2][:3] + [math.radians(350.0)] + BASE[2][4:]]
    code, _ = c.run(a, b)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(3, 0, 0)
    diffs = {int(r["a_index"]): float(r["ori_diff_deg"])
             for r in c.rows("matched.csv")}
    expected = {0: -1.0, 1: 1.0, 2: 0.0}
    c.check(all(abs(diffs.get(i, 999) - d) < 1e-3 for i, d in expected.items()),
            f"orientation differences (b - a) are {expected} (got {diffs})")
    oris = c.column("a_ori_deg") + c.column("b_ori_deg")
    c.check(all(0.0 <= o < 360.0 for o in oris),
            f"orientations are written in [0, 360) degrees (got {oris})")


def case_same_position(c):
    # SIFT gives one keypoint several orientations. They are paired by
    # orientation, whatever their order in the files.
    a = [BASE[0][:3] + [math.radians(ori)] + feature(0, 0, 0, 0, 10 + k)[4:]
         for k, ori in enumerate((10.0, 100.0, 200.0))]
    b = [a[2], a[0], a[1]]
    code, _ = c.run(a, b)
    c.check(code == 0, f"exit code 0 (got {code})")
    c.check_counts(3, 0, 0)
    c.check_close(c.column("ori_diff_deg"), 0.0, 1e-3, "orientation difference")
    c.check_close(c.column("desc_dist"), 0.0, 1e-6, "descriptor distance")


def case_max_avg_dist(c):
    # Each descriptor of b differs by 6 in one component: L2 distance 6.
    b = [f[:4] + [f[4] + 6.0] + f[5:] for f in BASE]
    code, output = c.run(BASE, b)
    c.check(code == 0, f"exit code 0 without a limit (got {code})")
    c.check("Average best-match descriptor distance: 6" in output,
            "average best-match descriptor distance 6 is printed")
    code, output = c.run(BASE, b, "--max-avg-dist", "7")
    c.check(code == 0, f"passes with --max-avg-dist 7 (got {code})")
    code, output = c.run(BASE, b, "--max-avg-dist", "5")
    c.check(code != 0 and "FAIL" in output,
            f"fails with --max-avg-dist 5 (exit code {code})")


CASES = {
    "identical": case_identical,
    "shifted": case_shifted,
    "missing": case_missing,
    "empty": case_empty,
    "orientation-wrap": case_orientation_wrap,
    "same-position": case_same_position,
    "max-avg-dist": case_max_avg_dist,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tool", required=True,
                        help="the popsift-compareSiftFiles binary")
    parser.add_argument("--work-dir", default=".",
                        help="where the feature and match files go")
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    c = Case(args.tool, args.work_dir, args.case)
    CASES[args.case](c)
    print(f"{args.case}: {'PASSED' if c.failures == 0 else 'FAILED'}"
          + (f", {c.failures} check(s) failed" if c.failures else ""))
    return 0 if c.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
