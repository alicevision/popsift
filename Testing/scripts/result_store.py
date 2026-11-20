"""Result stores: the SIFT results of one run, as files that any tool can
write and the comparison scripts read.

A store holds the results of one run, that is one SIFT tool with one
configuration, on one or more image sets:

    <stores dir>/<run name>/
        run.json                    the manifest
        <set>/<image stem>.sift     one feature file per image
        <set>/<image stem>.log      optional: the tool's output for the image
        <set>/<image stem>.skipped  written by collect_run.py when a PopSift
                                    run was skipped for lack of a GPU

<set> is the label of the image set, e.g. handcrafted or oxford/boat (a
folder per part), and <image stem> the image's path in the set without its
extension. A feature file holds one keypoint per line with 132 numbers, as
VLFeat's sift CLI writes them: x, y, sigma, orientation (radians) and 128
descriptor values in VLFeat's order, scaled to bytes. See
Testing/scripts/README.md.

This module also holds what the scripts share about the build's
popsift-testing-config.json and the image sets.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import time

# Written by CMake into the build directory (Testing/CMakeLists.txt).
TOOLS_FILE = "popsift-testing-config.json"

IMAGE_EXTENSIONS = (".pgm", ".ppm")
EXPECTATIONS_FILE = "expectations.json"

FORMAT = "popsift-result-store/1"
MANIFEST = "run.json"
FEATURES = ".sift"
LOG = ".log"
SKIPPED = ".skipped"

# Separates the prefix from the configuration in a run name.
PREFIX_SEPARATOR = "+"

# Numbers per line of a feature file, and of the Oxford affine format that
# is recognised only to be rejected.
VALUES_PER_LINE = 132
OXFORD_VALUES_PER_LINE = 133


class TestingError(Exception):
    """A setup problem that makes a run or a comparison impossible."""


# --- the build's configuration -----------------------------------------------

def load_tools(path):
    """Reads the popsift-testing-config.json that CMake generated."""
    try:
        with open(path, encoding="utf-8") as f:
            tools = json.load(f)
    except OSError as e:
        raise TestingError(f"cannot read {path}: {e.strerror}. "
                           "It is written when CMake configures with "
                           "-DBUILD_TESTING=ON.")
    for key in ("popsift_demo", "compare_sift_files", "vlfeat_sift",
                "paramsets", "image_sets"):
        if key not in tools:
            raise TestingError(f"{path} has no '{key}'")
    return tools


def find_tools_file(build_dir):
    """Finds popsift-testing-config.json in a build directory."""
    path = os.path.join(build_dir, TOOLS_FILE)
    if not os.path.exists(path):
        raise TestingError(f"no {TOOLS_FILE} in {build_dir}. Configure that "
                           "build with -DBUILD_TESTING=ON first.")
    return path


def stores_dir(tools, override=None):
    """Where the stores are: override, else stores_dir from the config,
    else stores/ next to the work dir."""
    if override:
        return os.path.abspath(override)
    if tools.get("stores_dir"):
        return tools["stores_dir"]
    work_dir = tools.get("work_dir") or os.getcwd()
    return os.path.join(os.path.dirname(os.path.normpath(work_dir)), "stores")


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._=+-]", "_", text)


# --- image sets --------------------------------------------------------------

def _configured_folder(tools, folder):
    """The label of the configured image set in folder, or None."""
    real = os.path.realpath(folder)
    for label, path in tools["image_sets"].items():
        if os.path.realpath(path) == real:
            return label
    return None


def resolve_images(tools, name_or_dir):
    """Returns (label, folder) for an image set label or a folder. The
    folder of a configured set gets that set's label."""
    sets = tools["image_sets"]
    if name_or_dir in sets:
        return name_or_dir, sets[name_or_dir]
    if os.path.isdir(name_or_dir):
        label = _configured_folder(tools, name_or_dir)
        if label:
            return label, sets[label]
        return os.path.basename(os.path.normpath(name_or_dir)), name_or_dir
    known = ", ".join(sorted(sets)) or "none"
    raise TestingError(f"'{name_or_dir}' is neither a configured image set "
                       f"nor a folder. Configured sets: {known}. Others come "
                       "from the folders below PopSift_TESTDATA_DIR, see "
                       "Testing/datasets/README.md.")


def expand_images(tools, name_or_dir):
    """The image sets that an --images argument names, as labels or
    folders: a configured label; a label prefix such as oxford for every
    configured set below it (oxford/bark, oxford/bikes, ...); a folder,
    which is given the label of the configured set there, or the labels of
    the configured sets below it; or another folder, labelled by its name.
    """
    sets = tools["image_sets"]
    prefix = name_or_dir.rstrip("/")
    labels = sorted(label for label in sets
                    if label == prefix or label.startswith(prefix + "/"))
    if labels:
        return labels
    if os.path.isdir(name_or_dir):
        label = _configured_folder(tools, name_or_dir)
        if label:
            labels = [label]
        else:
            real = os.path.realpath(name_or_dir) + os.sep
            labels = sorted(label for label, path in sets.items()
                            if os.path.realpath(path).startswith(real))
        if labels:
            print(f"-- {name_or_dir}: the configured image set(s) "
                  f"{', '.join(labels)}", flush=True)
            return labels
        print(f"-- {name_or_dir} is not a configured image set; its images "
              f"get the label '{resolve_images(tools, name_or_dir)[0]}'. "
              "Configure PopSift_TESTDATA_DIR to give sets the same labels "
              "in every build.", flush=True)
        return [name_or_dir]
    resolve_images(tools, name_or_dir)  # raises with the known sets
    return [name_or_dir]


def find_images(folder):
    """The .pgm/.ppm files in a folder, sorted. Its subfolders are image
    sets of their own."""
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXTENSIONS)
                  and os.path.isfile(os.path.join(folder, f)))


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
        raise TestingError(f"cannot read {path}: {e}")
    images = data.get("images", {})
    return {os.path.normpath(k): v for k, v in images.items()}


class ImageSet:
    """The images of a configured image set or of a folder."""

    def __init__(self, tools, images):
        self.name, self.folder = resolve_images(tools, images)
        self.expectations = load_expectations(self.folder)

    def files(self):
        files = find_images(self.folder)
        if not files:
            raise TestingError(f"no .pgm/.ppm images in {self.folder}")
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
            raise TestingError(f"{what} named '{key}' in {self.folder}")
        return found[0]

    def rel(self, src):
        return os.path.relpath(src, self.folder)

    def allow_empty(self, src):
        expect = self.expectations.get(os.path.normpath(self.rel(src)), {})
        return bool(expect.get("allow_empty"))


# --- run names ---------------------------------------------------------------

def check_prefix(prefix):
    """Fails unless prefix can start a run name."""
    if not prefix or safe_name(prefix) != prefix \
            or PREFIX_SEPARATOR in prefix or prefix.startswith("."):
        raise TestingError(f"the prefix '{prefix}' may only use letters, "
                           "digits and ._=- and must not start with '.'")


def run_name(configuration, prefix=None):
    """The name of a run: <prefix>+<configuration>, or the configuration
    alone, made safe for a folder name."""
    name = safe_name(configuration)
    if prefix:
        check_prefix(prefix)
        return f"{prefix}{PREFIX_SEPARATOR}{name}"
    return name


# --- feature files -----------------------------------------------------------

def read_features(path):
    """The keypoint lines of a feature file, without empty lines.

    Fails on a line that is not 132 numbers, and names the Oxford affine
    format (133 numbers per line), which the comparison does not support.
    """
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for number, line in enumerate(f, 1):
            fields = line.split()
            if not fields:
                continue
            try:
                [float(x) for x in fields]
            except ValueError:
                raise TestingError(f"{path}:{number}: not a line of numbers")
            if len(fields) == OXFORD_VALUES_PER_LINE:
                raise TestingError(
                    f"{path}:{number}: 133 numbers, the Oxford affine format "
                    "(x y a b c and 128 values). It has no orientation and "
                    "is not supported; convert it to VLFeat's format.")
            if len(fields) != VALUES_PER_LINE:
                raise TestingError(f"{path}:{number}: {len(fields)} numbers, "
                                   f"expected {VALUES_PER_LINE} (x y sigma "
                                   "orientation and 128 descriptor values)")
            lines.append(line if line.endswith("\n") else line + "\n")
    return lines


def write_sorted(lines, dest):
    """Writes feature lines sorted like 'sort -n': by x, then by the whole
    line. Returns the number of keypoints."""
    lines = sorted(lines, key=lambda line: (float(line.split()[0]), line))
    tmp = f"{dest}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.replace(tmp, dest)
    return len(lines)


# --- stores ------------------------------------------------------------------

def write_json(path, data):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


@contextlib.contextmanager
def locked(path, timeout=120.0):
    """Holds path + '.lock' while the block runs. CTest runs the images of
    a store in parallel, and each adds its set to the same run.json."""
    lock = path + ".lock"
    start = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() - start > timeout:
                raise TestingError(f"{lock} has been held for {timeout:g}s; "
                                   "remove it if no run is active")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        os.remove(lock)


class Store:
    """The result store of one run."""

    def __init__(self, path, manifest):
        self.path = path
        self.manifest = manifest

    @property
    def name(self):
        return self.manifest["name"]

    @property
    def label(self):
        return self.manifest.get("label") or self.name

    @classmethod
    def open(cls, path):
        manifest_path = os.path.join(path, MANIFEST)
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except OSError as e:
            raise TestingError(f"cannot read {manifest_path}: {e.strerror}")
        except ValueError as e:
            raise TestingError(f"{manifest_path}: {e}")
        if manifest.get("format") != FORMAT:
            raise TestingError(f"{manifest_path}: 'format' is not "
                               f"'{FORMAT}'")
        name = manifest.get("name")
        if not name or safe_name(name) != name:
            raise TestingError(f"{manifest_path}: 'name' must be letters, "
                               "digits and ._=+-")
        return cls(os.path.abspath(path), manifest)

    @classmethod
    def create(cls, path, manifest, image_set, replace=False):
        """Creates the store at path, or adds image_set to the one there.

        A store of another run, whose manifest differs in one of the fields
        named in SAME_RUN, is refused, or emptied first with replace.
        """
        os.makedirs(path, exist_ok=True)
        manifest_path = os.path.join(path, MANIFEST)
        with locked(manifest_path):
            other = None
            if os.path.exists(manifest_path):
                old = cls.open(path).manifest
                other = next((k for k in SAME_RUN
                              if old.get(k) != manifest.get(k)), None)
                if other and not replace:
                    raise TestingError(
                        f"{path} holds another run: its '{other}' is "
                        f"{old.get(other)!r}, this run's "
                        f"{manifest.get(other)!r}. Use another --prefix or "
                        "--replace.")
                if other:
                    print(f"-- Replacing the store {path}: its '{other}' "
                          "differs", flush=True)
                    _empty(path, keep=os.path.basename(manifest_path)
                           + ".lock")
            if os.path.exists(manifest_path) and not other:
                merged = dict(old)
                merged.update({k: v for k, v in manifest.items()
                               if k != "image_sets"})
                sets = list(old.get("image_sets", []))
            else:
                merged = dict(manifest)
                sets = []
            if image_set not in sets:
                sets.append(image_set)
            merged["image_sets"] = sorted(sets)
            write_json(manifest_path, merged)
        return cls(os.path.abspath(path), merged)

    def has_set(self, set_name):
        return os.path.isdir(os.path.join(self.path, *set_name.split("/")))

    def file(self, set_name, rel, ext=FEATURES):
        """The file of an image: rel is its path in the set."""
        stem = os.path.splitext(os.path.normpath(rel))[0]
        return os.path.join(self.path, *set_name.split("/"), stem + ext)

    def skipped(self, set_name, rel):
        """The reason the image was skipped, or None."""
        path = self.file(set_name, rel, SKIPPED)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip() or "skipped"


# A store refuses a run whose manifest differs in one of these.
SAME_RUN = ("name", "configuration", "command", "build")


def _empty(path, keep):
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if entry == keep:
            continue
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full)
        else:
            os.remove(full)


def find_store(name_or_path, stores):
    """The store at a path, or named so in the stores dir, or None."""
    for path in (name_or_path, os.path.join(stores, name_or_path)):
        if os.path.isfile(os.path.join(path, MANIFEST)):
            return Store.open(path)
    return None
