#!/usr/bin/env python3
"""Compare two SIFT runs on every image of a test image set.

A run is either 'vlfeat' (VLFeat's sift CLI) or the name of a parameter set
from Testing/paramsets.json, which is passed to popsift-demo.

For every image, both runs get their own work directory
(<work dir>/<set>/<image>/<run>/). The outputs are sorted, and
popsift-compareSiftFiles compares the first run against the second: it pairs
their features by position, scale and orientation, and writes matched.csv,
only_a.csv and only_b.csv, plus the descriptor nearest neighbour of every
feature in the other run (desc_nn_a.csv, desc_nn_b.csv). Every command is
printed, and each image ends with a one-line verdict. A report with
statistics per image and for the set (summary.csv, summary.json, report.md)
is written to <work dir>/<set>/report_<a>__vs__<b>/. With numpy and
matplotlib installed, each image and the report also get cdf.png, sorted.png
and descriptor_bins.png, and each image unmatched.png and image_heatmap.png.

Nothing fails on the statistics unless limits are given (--min-recall,
--min-precision, --max-desc-dist, --min-nn-agreement); without them an image
only fails if a run crashes, or exactly one run finds no features.

The paths to the binaries, the parameter sets and the image sets come from
the popsift-testing.json file that CMake writes into the build directory.

    popsift_compare.py --tools <build>/popsift-testing.json \\
                       --images handcrafted vlfeat-emulation vlfeat

--image limits this to one image. --step does one step of it, which is how
the CTest tests use the script: 'run' runs one run on one image, 'compare'
compares the earlier runs of one image, and 'report' writes the set report
from the images compared so far.

PopSift runs need a GPU; popsift-gpu-probe tells whether there is one. The
exit code is 0 if every image passed, 77 if nothing could be done for lack
of a GPU, and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import match_stats

_ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "analysis")

IMAGE_EXTENSIONS = (".pgm", ".ppm")

# The line popsift-compareSiftFiles prints to stderr at the end.
AVG_DIST_RE = re.compile(r"Average best-match descriptor distance:\s*(\S+)")


class CompareError(Exception):
    """A setup problem that makes a comparison impossible."""


# --- configuration ------------------------------------------------------------

def load_tools(path):
    """Reads the popsift-testing.json that CMake generated."""
    try:
        with open(path, encoding="utf-8") as f:
            tools = json.load(f)
    except OSError as e:
        raise CompareError(f"cannot read {path}: {e.strerror}. "
                           "It is written when CMake configures with "
                           "-DPopSift_TEST_WITH_VLFEAT=ON.")
    for key in ("popsift_demo", "compare_sift_files", "vlfeat_sift",
                "paramsets", "image_sets"):
        if key not in tools:
            raise CompareError(f"{path} has no '{key}'")
    return tools


def find_tools_file(build_dir, config=None):
    """Finds popsift-testing[-<config>].json in a build directory."""
    if config:
        candidates = [f"popsift-testing-{config}.json"]
    else:
        candidates = ["popsift-testing.json"]
        try:
            per_config = sorted(n for n in os.listdir(build_dir)
                                if re.fullmatch(r"popsift-testing-.+\.json", n))
        except OSError as e:
            raise CompareError(f"cannot list {build_dir}: {e.strerror}")
        if len(per_config) == 1:
            candidates.append(per_config[0])
        elif len(per_config) > 1 and not os.path.exists(
                os.path.join(build_dir, "popsift-testing.json")):
            names = ", ".join(n[len("popsift-testing-"):-len(".json")]
                              for n in per_config)
            raise CompareError(f"{build_dir} has several configurations "
                               f"({names}); choose one with --config")
    for name in candidates:
        path = os.path.join(build_dir, name)
        if os.path.exists(path):
            return path
    raise CompareError(f"no {candidates[0]} in {build_dir}. Configure that "
                       "build with -DPopSift_TEST_WITH_VLFEAT=ON first.")


def load_paramsets(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {name: entry for name, entry in data.items()
            if not name.startswith("_")}


def _option_name(arg):
    if not arg.startswith("--"):
        raise CompareError(f"parameter sets must use --name or --name=value, "
                           f"not '{arg}'")
    return arg[2:].split("=", 1)[0]


def resolve_paramset(paramsets, spec):
    """Returns the popsift-demo arguments of a set name with overrides."""
    name, *overrides = spec.split("@")
    if name not in paramsets or paramsets[name].get("abstract"):
        known = ", ".join(sorted(n for n, e in paramsets.items()
                                 if not e.get("abstract")))
        raise CompareError(f"unknown parameter set '{name}'. Known sets: "
                           f"vlfeat, {known}")

    def collect(n, seen):
        if n in seen:
            raise CompareError(f"parameter set '{n}' inherits from itself")
        entry = paramsets.get(n)
        if entry is None:
            raise CompareError(f"parameter set base '{n}' does not exist")
        options = collect(entry["base"], seen | {n}) if "base" in entry else {}
        for arg in entry.get("args", []):
            options[_option_name(arg)] = arg
        return options

    options = collect(name, frozenset())
    for o in overrides:
        if o.startswith("!"):
            options.pop(o[1:], None)
        else:
            options[_option_name("--" + o)] = "--" + o
    return list(options.values())


def resolve_images(tools, name_or_dir):
    """Returns (name, folder) for an image set name or a folder."""
    sets = tools["image_sets"]
    if name_or_dir in sets:
        return name_or_dir, sets[name_or_dir]
    if os.path.isdir(name_or_dir):
        return os.path.basename(os.path.normpath(name_or_dir)), name_or_dir
    known = ", ".join(sorted(sets)) or "none"
    raise CompareError(f"'{name_or_dir}' is neither a configured image set "
                       f"nor a folder. Configured sets: {known}. Others are "
                       "selected with PopSift_TEST_DATASETS, see "
                       "Testing/datasets/README.md.")


def find_images(folder):
    """All .pgm/.ppm files below a folder, sorted by relative path."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                found.append(os.path.join(root, f))
    return sorted(found, key=lambda p: os.path.relpath(p, folder))


EXPECTATIONS_FILE = "expectations.json"


def load_expectations(folder):
    """Per-image expectations of an image set, keyed by relative path.

    An image set folder may contain expectations.json, e.g.
        {"images": {"box-12x12-symm.pgm": {"allow_empty": true}}}
    allow_empty: the image may yield no features at all (in both runs).
    """
    path = os.path.join(folder, EXPECTATIONS_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise CompareError(f"cannot read {path}: {e}")
    images = data.get("images", {})
    return {os.path.normpath(k): v for k, v in images.items()}


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._=+-]", "_", text)


# --- images -------------------------------------------------------------------

def _read_pnm(path):
    """Reads a P2/P3/P5/P6 file. Returns (magic, width, height, maxval, data)."""
    with open(path, "rb") as f:
        raw = f.read()
    tokens = []
    pos = 0
    while len(tokens) < 4:
        while pos < len(raw) and raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b"#":
            while pos < len(raw) and raw[pos:pos + 1] not in (b"\n", b"\r"):
                pos += 1
            continue
        start = pos
        while pos < len(raw) and not raw[pos:pos + 1].isspace() \
                and raw[pos:pos + 1] != b"#":
            pos += 1
        if start == pos:
            raise CompareError(f"{path}: truncated PNM header")
        tokens.append(raw[start:pos])
    magic = tokens[0].decode("ascii", "replace")
    if magic not in ("P2", "P3", "P5", "P6"):
        raise CompareError(f"{path}: not a PGM or PPM file ({magic})")
    width, height, maxval = (int(t) for t in tokens[1:])
    channels = 3 if magic in ("P3", "P6") else 1
    count = width * height * channels
    if magic in ("P5", "P6"):
        pos += 1  # the single whitespace after maxval
        if maxval < 256:
            data = list(raw[pos:pos + count])
        else:
            b = raw[pos:pos + 2 * count]
            data = [(b[i] << 8) | b[i + 1] for i in range(0, len(b), 2)]
    else:
        data = [int(t) for t in raw[pos:].split()[:count]]
    if len(data) != count:
        raise CompareError(f"{path}: expected {count} samples, found {len(data)}")
    return magic, width, height, maxval, data


def to_gray_pgm(src, dest):
    """Writes src as an 8-bit binary PGM, the only input VLFeat's CLI reads.

    Colour is converted to Rec. 709 luma of the gamma-encoded values, rounded
    to the nearest integer. That is what ImageMagick 6's 'convert in.ppm
    out.pgm' does, which the old bash driver used, so results stay comparable.
    Both runs of a comparison always get the same converted file.
    """
    magic, width, height, maxval, data = _read_pnm(src)
    if magic in ("P3", "P6"):
        r, g, b = data[0::3], data[1::3], data[2::3]
        data = [int(0.212656 * x + 0.715158 * y + 0.072186 * z + 0.5)
                for x, y, z in zip(r, g, b)]
    if maxval != 255:
        data = [(v * 255 + maxval // 2) // maxval for v in data]
    tmp = f"{dest}.{os.getpid()}.tmp"  # runs of one image may convert at once
    with open(tmp, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (width, height))
        f.write(bytes(data))
    os.replace(tmp, dest)


def input_path(src, image_dir):
    """The binary 8-bit PGM that the runs read: src itself if it is one,
    else the converted copy in image_dir."""
    if src.lower().endswith(".pgm"):
        with open(src, "rb") as f:
            head = f.read(64)
        # P5 with maxval 255 is used as it is.
        fields = re.sub(rb"#[^\n]*\n", b" ", head).split()
        if len(fields) >= 4 and fields[0] == b"P5" and fields[3] == b"255":
            return src
    return os.path.join(image_dir, "input.pgm")


def prepare_input(src, image_dir):
    """Returns input_path(), converting src first if needed."""
    dest = input_path(src, image_dir)
    if dest != src and (not os.path.exists(dest)
                        or os.path.getmtime(dest) < os.path.getmtime(src)):
        print(f"-- Converting {os.path.basename(src)} to an 8-bit PGM", flush=True)
        to_gray_pgm(src, dest)
    return dest


# --- running ------------------------------------------------------------------

def run_command(cmd, cwd, log_path):
    """Runs cmd in cwd, prints it and writes its output to log_path."""
    print("   " + " ".join(_quote(c) for c in cmd), flush=True)
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    except OSError as e:
        raise CompareError(f"cannot run {cmd[0]}: {e.strerror}")
    seconds = time.monotonic() - start
    output = proc.stdout.decode("utf-8", "replace")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)
    return proc.returncode, output, seconds


def _quote(text):
    return f'"{text}"' if (" " in text or text == "") else text


def sort_features(src, dest):
    """Sorts feature lines like 'sort -n': by x, then by the whole line."""
    with open(src, encoding="utf-8", errors="replace") as f:
        lines = [line for line in f if line.strip()]

    def key(line):
        try:
            return (float(line.split()[0]), line)
        except ValueError:
            return (float("-inf"), line)

    lines.sort(key=key)
    with open(dest, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return len(lines)


class RunResult:
    def __init__(self, spec, features, count, seconds, workdir):
        self.spec = spec
        self.features = features    # sorted feature file
        self.count = count          # number of features
        self.seconds = seconds
        self.workdir = workdir


SORTED_FEATURES = "features-sorted.txt"
SKIPPED_MARKER = "skipped.txt"


def run_dir(image_dir, spec):
    return os.path.join(image_dir, safe_name(spec))


def load_run(spec, image_dir):
    """The RunResult of an earlier run on this image, or None."""
    workdir = run_dir(image_dir, spec)
    path = os.path.join(workdir, SORTED_FEATURES)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        count = sum(1 for line in f if line.strip())
    return RunResult(spec, path, count, None, workdir)


class Runner:
    """Runs vlfeat or popsift parameter sets on images, once per pair."""

    def __init__(self, tools):
        self.tools = tools
        self.paramsets = load_paramsets(tools["paramsets"])
        self._cache = {}
        self._gpu = None

    def check(self, specs, compare=True):
        """Fails early if a run cannot work, before any image is processed."""
        needed = {"compare_sift_files"} if compare else set()
        for spec in specs:
            if spec == "vlfeat":
                needed.add("vlfeat_sift")
            else:
                resolve_paramset(self.paramsets, spec)
                needed.update(("popsift_demo", "gpu_probe"))
        hints = {
            "vlfeat_sift": "cmake --build <build> --target vlfeat_sift",
            "popsift_demo": "cmake --build <build> --target popsift-demo",
            "gpu_probe": "cmake --build <build> --target popsift-gpu-probe",
            "compare_sift_files":
                "cmake --build <build> --target popsift-compareSiftFiles",
        }
        for key in sorted(needed):
            path = self.tools.get(key, "")
            if not os.path.isfile(path):
                raise CompareError(f"{path or key} does not exist. Build it "
                                   f"with: {hints[key]}")

    def gpu_available(self):
        """Runs popsift-gpu-probe once. Returns (usable, its message)."""
        if self._gpu is None:
            try:
                proc = subprocess.run([self.tools["gpu_probe"]],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
            except OSError as e:
                raise CompareError(f"cannot run {self.tools['gpu_probe']}: "
                                   f"{e.strerror}")
            message = proc.stdout.decode("utf-8", "replace").strip()
            if proc.returncode not in (0, SKIP):
                raise CompareError(f"popsift-gpu-probe failed with exit code "
                                   f"{proc.returncode}: {message}")
            self._gpu = (proc.returncode == 0, message)
        return self._gpu

    def run(self, spec, pgm, image_dir):
        """Runs spec on pgm. Returns a RunResult, None if the run failed,
        or SKIP if spec needs a GPU and there is none."""
        cache_key = (spec, pgm)
        if cache_key in self._cache:
            return self._cache[cache_key]
        workdir = run_dir(image_dir, spec)
        os.makedirs(workdir, exist_ok=True)
        # Nothing of an earlier run may be mistaken for this one's result.
        for name in (SORTED_FEATURES, SKIPPED_MARKER):
            if os.path.exists(os.path.join(workdir, name)):
                os.remove(os.path.join(workdir, name))
        print(f"-- {spec}", flush=True)
        if spec != "vlfeat":
            usable, message = self.gpu_available()
            if not usable:
                print(f"   skipped: {message}", flush=True)
                with open(os.path.join(workdir, SKIPPED_MARKER), "w",
                          encoding="utf-8") as f:
                    f.write(message + "\n")
                self._cache[cache_key] = SKIP
                return SKIP
        if spec == "vlfeat":
            raw = os.path.join(workdir, "vlfeat.sift")
            cmd = [self.tools["vlfeat_sift"], "--verbose",
                   f"--output={raw}", pgm]
        else:
            raw = os.path.join(workdir, "output-features.txt")
            cmd = [self.tools["popsift_demo"]] \
                + resolve_paramset(self.paramsets, spec) + ["-i", pgm]
        if os.path.exists(raw):
            os.remove(raw)
        code, output, seconds = run_command(cmd, workdir,
                                            os.path.join(workdir, "log.txt"))
        if code != 0 or not os.path.exists(raw):
            tail = "\n".join(output.splitlines()[-15:])
            result = None
            print(f"   failed with exit code {code}"
                  + ("" if os.path.exists(raw) else ", no feature file")
                  + (":\n" + tail if tail else ""), flush=True)
        else:
            sorted_path = os.path.join(workdir, SORTED_FEATURES)
            count = sort_features(raw, sorted_path)
            print(f"   {count} features in {seconds:.2f}s", flush=True)
            result = RunResult(spec, sorted_path, count, seconds, workdir)
        self._cache[cache_key] = result
        return result

    def compare(self, a, b, image_dir, max_avg_dist, allow_empty=False):
        """Compares run a against run b. Returns a CompareResult.

        allow_empty accepts two runs that both found no features.
        """
        outdir = clear_compare_dir(image_dir, a.spec, b.spec)
        print(f"-- compare {a.spec} against {b.spec}", flush=True)
        cmd = [self.tools["compare_sift_files"],
               "--max-avg-dist", str(max_avg_dist),
               "-o", os.path.join(outdir, "comparison.txt"),
               "-d", os.path.join(outdir, "descriptor-distances.txt"),
               "--match-dir", outdir,
               "--same-pose-px", str(match_stats.SAME_POSE_PX),
               "--same-pose-deg", str(match_stats.SAME_POSE_DEG)] \
            + (["--allow-empty"] if allow_empty else []) \
            + [a.features, b.features]
        code, output, _ = run_command(cmd, outdir,
                                      os.path.join(outdir, "log.txt"))
        match = AVG_DIST_RE.search(output)
        avg = float(match.group(1)) if match else None
        for line in output.splitlines():
            if line.startswith("FAIL"):
                print("   " + line, flush=True)
        stats = None
        if os.path.exists(os.path.join(outdir, "matched.csv")):
            stats = match_stats.image_stats(outdir)
        return CompareResult(code == 0, avg, outdir, stats)


# The files of popsift-compareSiftFiles --match-dir.
MATCH_FILES = ("matched.csv", "only_a.csv", "only_b.csv",
               "desc_nn_a.csv", "desc_nn_b.csv")


def compare_dir(image_dir, spec_a, spec_b):
    return os.path.join(image_dir,
                        f"{safe_name(spec_a)}__vs__{safe_name(spec_b)}")


def clear_compare_dir(image_dir, spec_a, spec_b):
    """Creates compare_dir() and removes the match files of an earlier
    comparison, which the set report would otherwise pick up."""
    outdir = compare_dir(image_dir, spec_a, spec_b)
    os.makedirs(outdir, exist_ok=True)
    for name in MATCH_FILES:
        if os.path.exists(os.path.join(outdir, name)):
            os.remove(os.path.join(outdir, name))
    return outdir


class CompareResult:
    def __init__(self, passed, avg, match_dir, stats):
        self.passed = passed          # the compare tool succeeded
        self.avg = avg                # average best-match descriptor distance
        self.match_dir = match_dir    # matched.csv, only_a.csv, only_b.csv
        self.stats = stats            # match_stats.image_stats(), or None


def load_plotting():
    """The optional plotting module, or None without numpy and matplotlib."""
    if _ANALYSIS_DIR not in sys.path:
        sys.path.insert(0, _ANALYSIS_DIR)
    try:
        import matplotlib
        matplotlib.use("Agg")  # only files are written, never a window
        import sift_plots
    except ImportError:
        return None
    return sift_plots


# --- one image set ------------------------------------------------------------

# Exit code of a step that could not run for lack of a GPU. CTest reports
# such tests as skipped (SKIP_RETURN_CODE).
SKIP = 77


class ImageSet:
    """The images of a set and where their outputs go."""

    def __init__(self, tools, images, work_dir):
        self.name, self.folder = resolve_images(tools, images)
        self.dir = os.path.join(work_dir,
                                safe_name(self.name.replace("/", "_")))
        self.expectations = load_expectations(self.folder)

    def files(self):
        files = find_images(self.folder)
        if not files:
            raise CompareError(f"no .pgm/.ppm images found under {self.folder}")
        return files

    def find(self, key):
        """The image named key: its path relative to the set, with or
        without extension, or its file name without extension."""
        files = self.files()
        found = [f for f in files
                 if key in (self.rel(f), os.path.splitext(self.rel(f))[0],
                            os.path.splitext(os.path.basename(f))[0])]
        if len(found) != 1:
            what = "no image" if not found else "several images"
            raise CompareError(f"{what} named '{key}' in {self.folder}")
        return found[0]

    def rel(self, src):
        return os.path.relpath(src, self.folder)

    def image_dir(self, src):
        """The work directory of one image, named after it."""
        return os.path.join(self.dir,
                            safe_name(os.path.splitext(self.rel(src))[0]))

    def allow_empty(self, src):
        expect = self.expectations.get(os.path.normpath(self.rel(src)), {})
        return bool(expect.get("allow_empty"))

    def report_dir(self, run_a, run_b):
        return os.path.join(self.dir,
                            f"report_{safe_name(run_a)}__vs__{safe_name(run_b)}")


def compare_image(runner, imgset, src, a, b, max_avg_dist, limits, plotting):
    """Compares the runs a and b of one image and prints the verdict.

    Returns the report entry of the image (None if the comparison produced
    no statistics) and whether it passed.
    """
    rel = imgset.rel(src)
    image_dir = imgset.image_dir(src)
    pgm = input_path(src, image_dir)
    res = runner.compare(a, b, image_dir, max_avg_dist, imgset.allow_empty(src))
    problems = []
    entry = None
    if res.stats is not None:
        problems = match_stats.check_limits(res.stats, **limits)
        entry = {"name": rel, "input": pgm, "match_dir": res.match_dir,
                 "stats": res.stats}
        if plotting is not None:
            plotting.render_unmatched(
                pgm, res.match_dir,
                os.path.join(res.match_dir, "unmatched.png"),
                label_a=a.spec, label_b=b.spec)
            title = f"{rel}: a = {a.spec}, b = {b.spec}"
            plotting.render_cdfs(os.path.join(res.match_dir, "cdf.png"),
                                 title, [(rel, res.stats["_rows"])])
            render_differences(plotting, res.match_dir, title,
                               res.stats["_rows"],
                               [plotting.read_desc_bins(res.match_dir)])
            plotting.render_image_heatmap(
                pgm, plotting.read_matches(res.match_dir),
                os.path.join(res.match_dir, "image_heatmap.png"), title)
    passed = res.passed and not problems
    verdict = "PASSED" if passed else "FAILED"
    detail = match_stats.one_line(res.stats) if res.stats else \
        f"{a.count} vs {b.count} features"
    print(f"{verdict} {rel}: {detail}")
    for p in problems:
        print(f"   limit: {p}")
    if not passed:
        print(f"   details: {res.match_dir}")
    return entry, passed


def render_differences(plotting, out_dir, title, rows, desc_bins):
    """Writes sorted.png and, if desc_bins.csv exists, descriptor_bins.png
    (the plots of the 2021 comparison scripts) into out_dir."""
    plotting.render_sorted(os.path.join(out_dir, "sorted.png"), title, rows)
    bins = plotting.merge_desc_bins(desc_bins)
    if bins:
        plotting.render_desc_bins(
            os.path.join(out_dir, "descriptor_bins.png"), title, bins)


def write_set_report(imgset, run_a, run_b, reported, plotting):
    """Writes the report of the set over the reported images."""
    report_dir = imgset.report_dir(run_a, run_b)
    total = match_stats.set_stats([r["stats"] for r in reported])
    if plotting is not None:
        os.makedirs(report_dir, exist_ok=True)
        title = (f"{imgset.name}, {len(reported)} image(s): a = {run_a}, "
                 f"b = {run_b}")
        plotting.render_cdfs(os.path.join(report_dir, "cdf.png"), title,
                             [(imgset.name, total["_rows"])])
        render_differences(plotting, report_dir, title, total["_rows"],
                           [plotting.read_desc_bins(r["match_dir"])
                            for r in reported])
    match_stats.write_report(report_dir, run_a, run_b, imgset.name, reported,
                             total)
    print(f"Set: {match_stats.one_line(total)}")
    print(f"Report: {report_dir}/report.md (also summary.csv, summary.json)")


def plotting_or_note(plots):
    plotting = load_plotting() if plots else None
    if plots and plotting is None:
        print("Note: install numpy and matplotlib (Testing/analysis/"
              "requirements.txt) to get the plots.")
    return plotting


def compare_image_set(tools, images, run_a, run_b, work_dir, max_avg_dist,
                      limits=None, plots=True, only=None):
    """Runs and compares a and b on every image of a set, or only on the
    image named only. Writes the set report."""
    imgset = ImageSet(tools, images, work_dir)
    runner = Runner(tools)
    runner.check([run_a, run_b])
    files = [imgset.find(only)] if only else imgset.files()
    print(f"Comparing {run_a} against {run_b} on {len(files)} image(s) "
          f"of '{imgset.name}' in {imgset.folder}")
    print(f"Work directory: {imgset.dir}")
    limits = limits or {}
    plotting = plotting_or_note(plots)

    failed = []
    skipped = []
    reported = []
    for index, src in enumerate(files, 1):
        rel = imgset.rel(src)
        print(f"\n==================== [{index}/{len(files)}] {rel} "
              "====================", flush=True)
        image_dir = imgset.image_dir(src)
        os.makedirs(image_dir, exist_ok=True)
        try:
            pgm = prepare_input(src, image_dir)
        except CompareError as e:
            print(f"FAILED {rel}: {e}")
            failed.append(rel)
            continue
        a = runner.run(run_a, pgm, image_dir)
        b = runner.run(run_b, pgm, image_dir)
        if SKIP in (a, b):
            print(f"SKIPPED {rel}: no usable GPU")
            skipped.append(rel)
            continue
        if a is None or b is None:
            print(f"FAILED {rel}: a run did not produce features "
                  f"(see {image_dir})")
            failed.append(rel)
            continue
        entry, passed = compare_image(runner, imgset, src, a, b, max_avg_dist,
                                      limits, plotting)
        if entry is not None:
            reported.append(entry)
        if not passed:
            failed.append(rel)

    print("\n==================== Summary ====================")
    if reported:
        write_set_report(imgset, run_a, run_b, reported, plotting)
    done = len(files) - len(skipped)
    if done:
        print(f"{done - len(failed)} of {done} image(s) passed"
              + (f", {len(skipped)} skipped for lack of a GPU" if skipped else ""))
    else:
        print(f"All {len(skipped)} image(s) skipped for lack of a GPU")
    for rel in failed:
        print(f"  failed: {rel}")
    if failed:
        print(f"Details are in the per-image folders under {imgset.dir}")
        return 1
    return SKIP if skipped and not done else 0


# --- single steps, one CTest test each ----------------------------------------

def step_run(tools, images, key, spec, work_dir):
    """Runs one spec on one image."""
    imgset = ImageSet(tools, images, work_dir)
    runner = Runner(tools)
    runner.check([spec], compare=False)
    src = imgset.find(key)
    rel = imgset.rel(src)
    image_dir = imgset.image_dir(src)
    os.makedirs(image_dir, exist_ok=True)
    print(f"Running {spec} on {rel} of '{imgset.name}'")
    print(f"Work directory: {run_dir(image_dir, spec)}")
    result = runner.run(spec, prepare_input(src, image_dir), image_dir)
    if result is SKIP:
        print(f"SKIPPED {spec} on {rel}: no usable GPU")
        return SKIP
    if result is None:
        print(f"FAILED {spec} on {rel}: no features written, see "
              f"{os.path.join(run_dir(image_dir, spec), 'log.txt')}")
        return 1
    print(f"PASSED {spec} on {rel}: {result.count} features in "
          f"{result.seconds:.2f}s")
    return 0


def step_compare(tools, images, key, run_a, run_b, work_dir, max_avg_dist,
                 limits, plots):
    """Compares the earlier runs a and b of one image."""
    imgset = ImageSet(tools, images, work_dir)
    runner = Runner(tools)
    runner.check([], compare=True)
    src = imgset.find(key)
    rel = imgset.rel(src)
    image_dir = imgset.image_dir(src)
    print(f"Comparing {run_a} against {run_b} on {rel} of '{imgset.name}'")
    clear_compare_dir(image_dir, run_a, run_b)
    runs = []
    for spec in (run_a, run_b):
        run = load_run(spec, image_dir)
        if run is None:
            workdir = run_dir(image_dir, spec)
            if os.path.exists(os.path.join(workdir, SKIPPED_MARKER)):
                print(f"SKIPPED {rel}: the {spec} run was skipped for lack "
                      "of a GPU")
                return SKIP
            print(f"FAILED {rel}: the {spec} run left no features in "
                  f"{workdir}; it failed or did not run")
            return 1
        runs.append(run)
    _, passed = compare_image(runner, imgset, src, runs[0], runs[1],
                              max_avg_dist, limits, plotting_or_note(plots))
    return 0 if passed else 1


def step_report(tools, images, run_a, run_b, work_dir, plots):
    """Writes the set report from the image comparisons that exist."""
    imgset = ImageSet(tools, images, work_dir)
    files = find_images(imgset.folder)
    if not files:
        # The download of the set failed, which its own test reports.
        print(f"SKIPPED: no images in {imgset.folder}")
        return SKIP
    reported = []
    missing = []
    for src in files:
        image_dir = imgset.image_dir(src)
        match_dir = compare_dir(image_dir, run_a, run_b)
        if not os.path.exists(os.path.join(match_dir, "matched.csv")):
            missing.append(imgset.rel(src))
            continue
        reported.append({"name": imgset.rel(src),
                         "input": input_path(src, image_dir),
                         "match_dir": match_dir,
                         "stats": match_stats.image_stats(match_dir)})
    print(f"Report of {run_a} against {run_b} on '{imgset.name}': "
          f"{len(reported)} compared image(s)")
    for rel in missing:
        print(f"  not compared: {rel}")
    if not reported:
        print("SKIPPED: no image of the set was compared")
        return SKIP
    write_set_report(imgset, run_a, run_b, reported, plotting_or_note(plots))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Runs are 'vlfeat' or a parameter set from Testing/paramsets.json, "
               "optionally with overrides such as loop@gauss-mode=vlfeat.")
    parser.add_argument("--tools", required=True,
                        help="popsift-testing.json from the build directory")
    parser.add_argument("--images", required=True,
                        help="a configured image set (e.g. handcrafted, "
                             "oxford/boat) or a folder of .pgm/.ppm files")
    parser.add_argument("--image",
                        help="only this image of the set: its path in the "
                             "set, with or without extension")
    parser.add_argument("--step", choices=("run", "compare", "report"),
                        help="do one step only: 'run' one run on --image, "
                             "'compare' the earlier runs a and b of --image, "
                             "or 'report' on the compared images of the set. "
                             "Default: all of them")
    parser.add_argument("--work-dir",
                        help="where outputs go (default: work_dir from --tools)")
    parser.add_argument("--max-avg-dist", type=float, default=-1.0,
                        help="fail an image whose average best-match descriptor "
                             "distance exceeds this; negative only reports "
                             "(default: %(default)s)")
    parser.add_argument("--min-recall", type=float, default=-1.0,
                        help="fail an image where fewer than this share of "
                             "run_b's features are matched (negative: off)")
    parser.add_argument("--min-precision", type=float, default=-1.0,
                        help="fail an image where fewer than this share of "
                             "run_a's features are matched (negative: off)")
    parser.add_argument("--max-desc-dist", type=float, default=-1.0,
                        help="fail an image whose median descriptor distance "
                             "of matched pairs exceeds this (negative: off)")
    parser.add_argument("--min-nn-agreement", type=float, default=-1.0,
                        help="fail an image where fewer than this share of "
                             "the matched features of either run have their "
                             "geometric match as descriptor nearest "
                             "neighbour (negative: off)")
    parser.add_argument("--no-plots", action="store_true",
                        help="do not write the plots")
    parser.add_argument("runs", nargs="+", metavar="run",
                        help="the run under test (a) and the reference it is "
                             "compared with (b); only one run with --step run")
    args = parser.parse_args(argv)

    wanted = 1 if args.step == "run" else 2
    if len(args.runs) != wanted:
        parser.error(f"expected {wanted} run(s), got {len(args.runs)}")
    if args.step in ("run", "compare") and not args.image:
        parser.error(f"--step {args.step} needs --image")

    try:
        tools = load_tools(args.tools)
        work_dir = args.work_dir or tools.get("work_dir") or os.getcwd()
        limits = {"min_recall": args.min_recall,
                  "min_precision": args.min_precision,
                  "max_desc_dist": args.max_desc_dist,
                  "min_nn_agreement": args.min_nn_agreement}
        plots = not args.no_plots
        if args.step == "run":
            return step_run(tools, args.images, args.image, args.runs[0],
                            work_dir)
        if args.step == "compare":
            return step_compare(tools, args.images, args.image, *args.runs,
                                work_dir, args.max_avg_dist, limits, plots)
        if args.step == "report":
            return step_report(tools, args.images, *args.runs, work_dir,
                               plots)
        return compare_image_set(tools, args.images, *args.runs, work_dir,
                                 args.max_avg_dist, limits, plots, args.image)
    except CompareError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
