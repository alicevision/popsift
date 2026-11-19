#!/usr/bin/env python3
"""Run one SIFT configuration on image sets and keep the results as a store.

A configuration is 'vlfeat' (VLFeat's sift CLI) or the name of a parameter
set from Testing/paramsets.json, which is passed to popsift-demo, optionally
with overrides such as loop@gauss-mode=vlfeat. The results go into the
result store <stores dir>/<run name>/, where the run name is
<prefix>+<configuration>, or the configuration alone without --prefix:

    collect_run.py --tools <build>/popsift-testing-config.json \\
                   --prefix atan2-fast --images handcrafted \\
                   --images oxford/boat popsift-vlfeat

writes <build>/Testing/stores/atan2-fast+popsift-vlfeat/. A store holds
run.json, which describes the run (tool, command, build options, GPU, host,
time), and per image <set>/<image stem>.sift with the keypoints and
descriptors, and <set>/<image stem>.log with the tool's output.
popsift_compare.py compares two stores; the store layout is described in
Testing/scripts/README.md, so that other SIFT tools can write stores too.

Collecting into an existing store adds or replaces images and image sets,
but only for the same run: a store of another configuration, command or
build is refused, or emptied first with --replace. The CTest tests and
popsift_compare.py use --replace for the stores they collect, so that a
reconfigured build does not fail on the stores of its earlier
configuration.

This is the only script that runs popsift-demo or VLFeat. PopSift runs need
a GPU; popsift-gpu-probe tells whether there is one. The exit code is 0 if
every image produced features, 77 if nothing could be done for lack of a
GPU, and 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
# Runs the configured test binaries, never through a shell.
import subprocess  # nosec B404
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import result_store as rs  # noqa: E402
from result_store import TestingError  # noqa: E402

# Exit code of a run that could not happen for lack of a GPU. CTest reports
# such tests as skipped (SKIP_RETURN_CODE).
SKIP = 77


# --- parameter sets ----------------------------------------------------------

def load_paramsets(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    paramsets = {name: entry for name, entry in data.items()
                 if not name.startswith("_")}
    for name in paramsets:
        if rs.PREFIX_SEPARATOR in name:
            raise TestingError(f"{path}: the parameter set '{name}' contains "
                               f"'{rs.PREFIX_SEPARATOR}', which separates "
                               "prefixes from configurations in run names")
    return paramsets


def print_list(tools):
    """Prints the configurations and image sets that tools knows about."""
    paramsets = load_paramsets(tools["paramsets"])
    print("Parameter sets (Testing/paramsets.json):")
    print(f"  {'vlfeat':18} VLFeat's sift CLI")
    for name, entry in sorted(paramsets.items()):
        if not entry.get("abstract"):
            print(f"  {name:18} {entry.get('description', '')}")
    print("Image sets:")
    for name, folder in sorted(tools["image_sets"].items()):
        print(f"  {name:18} {folder}")


def _option_name(arg):
    if not arg.startswith("--"):
        raise TestingError(f"parameter sets must use --name or --name=value, "
                           f"not '{arg}'")
    return arg[2:].split("=", 1)[0]


def resolve_paramset(paramsets, spec):
    """Returns the popsift-demo arguments of a set name with overrides."""
    name, *overrides = spec.split("@")
    if name not in paramsets or paramsets[name].get("abstract"):
        known = ", ".join(sorted(n for n, e in paramsets.items()
                                 if not e.get("abstract")))
        raise TestingError(f"unknown parameter set '{name}'. Known sets: "
                           f"vlfeat, {known}")

    def collect(n, seen):
        if n in seen:
            raise TestingError(f"parameter set '{n}' inherits from itself")
        entry = paramsets.get(n)
        if entry is None:
            raise TestingError(f"parameter set base '{n}' does not exist")
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


def is_configuration(tools, spec):
    """Whether spec is 'vlfeat' or a parameter set, possibly with
    overrides."""
    if spec == "vlfeat":
        return True
    if rs.PREFIX_SEPARATOR in spec or os.sep in spec:
        return False
    try:
        resolve_paramset(load_paramsets(tools["paramsets"]), spec)
    except TestingError:
        return False
    return True


# --- images ------------------------------------------------------------------

def _read_pnm(path):
    """Reads a P2/P3/P5/P6 file.

    Returns (magic, width, height, maxval, data).
    """
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
            raise TestingError(f"{path}: truncated PNM header")
        tokens.append(raw[start:pos])
    magic = tokens[0].decode("ascii", "replace")
    if magic not in ("P2", "P3", "P5", "P6"):
        raise TestingError(f"{path}: not a PGM or PPM file ({magic})")
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
        raise TestingError(f"{path}: expected {count} samples, "
                           f"found {len(data)}")
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
        os.makedirs(image_dir, exist_ok=True)
        print(f"-- Converting {os.path.basename(src)} to an 8-bit PGM",
              flush=True)
        to_gray_pgm(src, dest)
    return dest


# --- running -----------------------------------------------------------------

def run_command(cmd, cwd, log_path):
    """Runs cmd in cwd, prints it and writes its output to log_path."""
    print("   " + " ".join(_quote(c) for c in cmd), flush=True)
    start = time.monotonic()
    try:
        # cmd is a binary from popsift-testing-config.json with arguments
        # built here, run without a shell.
        proc = subprocess.run(cmd, cwd=cwd,  # nosec B603
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
    except OSError as e:
        raise TestingError(f"cannot run {cmd[0]}: {e.strerror}")
    seconds = time.monotonic() - start
    output = proc.stdout.decode("utf-8", "replace")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)
    return proc.returncode, output, seconds


def _quote(text):
    return f'"{text}"' if (" " in text or text == "") else text


HINTS = {
    "vlfeat_sift": "cmake --build <build> --target vlfeat_sift",
    "popsift_demo": "cmake --build <build> --target popsift-demo",
    "gpu_probe": "cmake --build <build> --target popsift-gpu-probe",
    "compare_sift_files":
        "cmake --build <build> --target popsift-compareSiftFiles",
}


def check_binaries(tools, keys):
    """Fails early if a binary is missing, before any image is processed."""
    for key in sorted(keys):
        path = tools.get(key, "")
        if not os.path.isfile(path):
            raise TestingError(f"{path or key} does not exist. Build it "
                               f"with: {HINTS[key]}")


class Collector:
    """Runs one configuration on images and writes the results to a store."""

    def __init__(self, tools, configuration, prefix=None, label=None,
                 stores=None, replace=False):
        self.tools = tools
        self.configuration = configuration
        self.name = rs.run_name(configuration, prefix)
        self.label = label
        self.replace = replace
        self.stores = rs.stores_dir(tools, stores)
        self.path = os.path.join(self.stores, self.name)
        if configuration == "vlfeat":
            self.args = []
            check_binaries(tools, ["vlfeat_sift"])
        else:
            self.args = resolve_paramset(load_paramsets(tools["paramsets"]),
                                         configuration)
            check_binaries(tools, ["popsift_demo", "gpu_probe"])
        self._gpu = None

    def gpu_available(self):
        """Runs popsift-gpu-probe once. Returns (usable, its message)."""
        if self._gpu is None:
            try:
                proc = subprocess.run([self.tools["gpu_probe"]],  # nosec B603
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, check=False)
            except OSError as e:
                raise TestingError(f"cannot run {self.tools['gpu_probe']}: "
                                   f"{e.strerror}")
            message = proc.stdout.decode("utf-8", "replace").strip()
            if proc.returncode not in (0, SKIP):
                raise TestingError(f"popsift-gpu-probe failed with exit code "
                                   f"{proc.returncode}: {message}")
            self._gpu = (proc.returncode == 0, message)
        return self._gpu

    def command(self, pgm, output):
        """The command that writes the features of pgm to output, which is
        in the command's working directory."""
        if self.configuration == "vlfeat":
            return [self.tools["vlfeat_sift"], "--verbose",
                    f"--output={output}", pgm]
        return [self.tools["popsift_demo"]] + self.args + ["-i", pgm]

    def manifest(self, image_set):
        # VLFeat does not depend on PopSift's build options.
        build = {} if self.configuration == "vlfeat" \
            else dict(self.tools.get("build", {}))
        manifest = {
            "format": rs.FORMAT,
            "name": self.name,
            "configuration": self.configuration,
            "tool": "vlfeat sift" if self.configuration == "vlfeat"
                    else "popsift-demo",
            "command": " ".join(_quote(c) for c in self.command(
                "{image}", "{output}")),
            "version": build.pop("version", ""),
            "build": build,
            "host": platform.node(),
            "created": datetime.datetime.now(datetime.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "image_sets": [image_set],
        }
        if self.label:
            manifest["label"] = self.label
        if self.configuration != "vlfeat":
            manifest["device"] = self.gpu_available()[1]
        return manifest

    def image(self, store, imgset, src):
        """Runs on one image. Returns the number of features, None if the
        run failed, or SKIP."""
        rel = imgset.rel(src)
        features = store.file(imgset.name, rel)
        log = store.file(imgset.name, rel, rs.LOG)
        skipped = store.file(imgset.name, rel, rs.SKIPPED)
        os.makedirs(os.path.dirname(features), exist_ok=True)
        # Nothing of an earlier run may be mistaken for this one's result.
        for path in (features, log, skipped):
            if os.path.exists(path):
                os.remove(path)
        print(f"-- {self.name}: {rel}", flush=True)
        if self.configuration != "vlfeat":
            usable, message = self.gpu_available()
            if not usable:
                print(f"   skipped: {message}", flush=True)
                with open(skipped, "w", encoding="utf-8") as f:
                    f.write(message + "\n")
                return SKIP
        pgm = prepare_input(src, os.path.join(
            self.stores, ".inputs", *imgset.name.split("/"),
            rs.safe_name(os.path.splitext(rel)[0])))
        workdir = tempfile.mkdtemp(prefix=".run-", dir=store.path)
        try:
            raw = os.path.join(workdir, "vlfeat.sift"
                               if self.configuration == "vlfeat"
                               else "output-features.txt")
            code, output, seconds = run_command(
                self.command(pgm, raw), workdir, log)
            if code != 0 or not os.path.exists(raw):
                tail = "\n".join(output.splitlines()[-15:])
                print(f"   failed with exit code {code}"
                      + ("" if os.path.exists(raw) else ", no feature file")
                      + (":\n" + tail if tail else ""), flush=True)
                return None
            try:
                count = rs.write_sorted(rs.read_features(raw), features)
            except TestingError as e:
                print(f"   failed: {e}", flush=True)
                return None
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        print(f"   {count} features in {seconds:.2f}s", flush=True)
        return count

    def collect(self, images, only=None):
        """Runs on every image of the image set images, or only on the image
        named only. Returns {relative path: count, None or SKIP}."""
        imgset = rs.ImageSet(self.tools, images)
        files = [imgset.find(only)] if only else imgset.files()
        store = rs.Store.create(self.path, self.manifest(imgset.name),
                                imgset.name, self.replace)
        print(f"Collecting {self.name} on {len(files)} image(s) of "
              f"'{imgset.name}' into {store.path}", flush=True)
        return {imgset.rel(src): self.image(store, imgset, src)
                for src in files}


def summarize(results):
    """Prints how the images went. Returns the exit code."""
    values = list(results.values())
    failed = [rel for rel, r in results.items() if r is None]
    skipped = sum(1 for r in values if r is SKIP)
    done = len(values) - skipped
    if done:
        print(f"{done - len(failed)} of {done} image(s) produced features"
              + (f", {skipped} skipped for lack of a GPU" if skipped else ""))
    else:
        print(f"All {skipped} image(s) skipped for lack of a GPU")
    for rel in failed:
        print(f"  failed: {rel}")
    if failed:
        return 1
    return SKIP if skipped and not done else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    parser.add_argument("--tools", required=True,
                        help="popsift-testing-config.json from the build "
                             "directory")
    parser.add_argument("--list", action="store_true",
                        help="list the parameter sets and image sets, and "
                             "exit")
    parser.add_argument("--images", action="append", default=[],
                        help="a configured image set (e.g. handcrafted, "
                             "oxford/boat), the start of the labels of "
                             "several (oxford), or a folder of .pgm/.ppm "
                             "files; may be given several times")
    parser.add_argument("--image",
                        help="only this image of the (single) set: its path "
                             "in the set, with or without extension")
    parser.add_argument("--prefix",
                        help="the run name is <prefix>+<configuration>")
    parser.add_argument("--label",
                        help="how reports and the explorer call the run "
                             "(default: its name)")
    parser.add_argument("--stores",
                        help="where the stores are (default: stores_dir "
                             "from --tools)")
    parser.add_argument("--replace", action="store_true",
                        help="replace a store of the same name that holds "
                             "another run (another configuration, command "
                             "or build), instead of refusing it")
    parser.add_argument("configuration", nargs="?",
                        help="'vlfeat' or a parameter set, optionally with "
                             "overrides such as loop@gauss-mode=vlfeat")
    args = parser.parse_intermixed_args(argv)

    try:
        tools = rs.load_tools(args.tools)
        if args.list:
            print_list(tools)
            return 0
        if not args.configuration or not args.images:
            parser.error("a configuration and --images are required "
                         "(or use --list)")
        image_sets = [s for images in args.images
                      for s in rs.expand_images(tools, images)]
        if args.image and len(image_sets) != 1:
            parser.error(f"--image needs a single image set, not "
                         f"{', '.join(image_sets)}")
        collector = Collector(tools, args.configuration, args.prefix,
                              args.label, args.stores, args.replace)
        results = {}
        for images in dict.fromkeys(image_sets):
            for rel, r in collector.collect(images, args.image).items():
                results[f"{images}: {rel}"] = r
        print(f"Store: {collector.path}")
        return summarize(results)
    except TestingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
