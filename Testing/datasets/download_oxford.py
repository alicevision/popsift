#!/usr/bin/env python3
"""Download Oxford affine-covariant region sets for PopSift tests.

Usage:
    python3 download_oxford.py <dir> [set ...]

The script accepts a required test-data directory and zero or more dataset
names. With no names, it processes all eight supported sets. It writes each
set under ``<dir>/oxford/<set>`` and skips a set when that directory already
contains a ``.pgm`` or ``.ppm`` file. Each downloaded tar archive is checked
against the SHA256 digest in ``SHA256`` before extraction.

Supported sets:
    bark, bikes, boat, graf, leuven, trees, ubc, and wall.

The Oxford Visual Geometry Group publishes these research datasets. They are
not redistributed with PopSift.
"""
import argparse
import hashlib
import io
import os
import sys
import tarfile
import urllib.request

URL: str = ("https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/{}.tar.gz")
SHA256: dict[str, str] = {
    "bark": "f631c09a0f43fc1e092203f32f8f27e2e43e59b524918255d45474c50c91ea4f",
    "bikes": "3b95c49accadfd71e63c2fa0f7602abcbadd87d7f53ff54eb1efb15210d19e0c",
    "boat": "6e5721f37fbf9c3e974fd05935a42c1670f3e71aa3e5b218807ea046a068c603",
    "graf": "999871b945ee968a00a0d5f9af957d1382fb9dae1511cdee9553366817b53b5b",
    "leuven": "9b1ade6e64e5b27eee43463ff8f9ad2bd2f9c6911666e93de2634f3a5c0fe5dc",
    "trees": "04a39f201bffbc5035a560136fb37d2b36ecaa1532113eebfb113c90b5327f5d",
    "ubc": "26ab4ac94c8d0b2ded66c3bfaeb3f12589ec33005e5b4d5fcba4639f5b6e5414",
    "wall": "3cb3df75f40471f093adf6647d13df6a9f4312f7a222fc591d03f6c08e086344",
}


def has_images(folder: str) -> bool:
    """Return whether a folder contains an image file.

    The folder must be an existing directory. A file directly inside it is
    considered an image when its name, compared case-insensitively, ends with
    ``.pgm`` or ``.ppm``. Subdirectories and files in nested directories are
    not inspected.

    Args:
        folder: Directory path to inspect.

    Returns:
        True if ``folder`` is a directory and at least one direct child has a
        ``.pgm`` or ``.ppm`` suffix; otherwise, False.

    Raises:
        OSError: If listing the existing directory fails.
    """
    return os.path.isdir(folder) and any(
        f.lower().endswith((".pgm", ".ppm")) for f in os.listdir(folder))


def download(name: str, dest: str, url: str) -> None:
    """Download, verify, and extract one Oxford dataset archive.

    This function performs one HTTP or HTTPS download with a 120-second
    timeout. It computes SHA256 over the complete response bytes and compares
    that digest with ``SHA256[name]``. A mismatch causes process termination
    before any destination directory or extracted file is created.

    On success, it creates ``dest`` and extracts only regular-file members.
    Archive directories and all archive-relative directory components are
    ignored: each regular file is written directly below ``dest`` using only
    its basename. It does not extract symbolic links, directories, or other
    special archive members.

    Args:
        name: Supported dataset name used to select the expected SHA256.
            It must be a key in ``SHA256``.
        dest: Destination directory for flattened extracted image files.
        url: Complete HTTP or HTTPS archive URL. The caller is responsible
            for substituting ``{}`` in ``URL`` if a template is used.

    Side Effects:
        Prints ``"<name>: downloading <url>"`` before downloading and
        ``"<name>: unpacked into <dest>"`` after successful extraction.
        Creates ``dest`` and writes regular files into it.

    Raises:
        SystemExit: With status string and exit status 1 if the download
            fails, the digest is wrong, ``name`` is unknown, or a regular
            archive member cannot provide an extraction stream.
        tarfile.ReadError: If the downloaded bytes are not a readable tar
            archive.
        OSError: If creating ``dest`` or writing extracted files fails.

    Examples:
        The name ``"boat"`` requires the digest in ``SHA256["boat"]`` and
        writes regular archive files directly into ``dest``.
    """
    print(f"{name}: downloading {url}", flush=True)
    try:
	# main() accepts only http and https URLs.
        with urllib.request.urlopen(url, timeout=120) as resp:  # nosec B310
            data = resp.read()
    except OSError as e:
        sys.exit(f"{name}: cannot download {url}: {e}")
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        sys.exit(f"{name}: the archive does not have the expected SHA256")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            if member.isfile():  # unpack flat, never outside dest
                with open(os.path.join(dest, os.path.basename(member.name)),
                          "wb") as f:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        sys.exit(f"{name}: cannot unpack {member.name}")
                    f.write(extracted.read())
    print(f"{name}: unpacked into {dest}")


def main() -> None:
    """Parse command-line arguments and download the requested datasets.

    Command-line contract:
        ``dir`` is required and is the test-data root. ``sets`` is optional;
        with no set names, all keys in ``SHA256`` are processed in sorted
        order. ``--url`` defaults to ``URL`` and must be an HTTP or HTTPS URL
        containing at least no validation assumption; ``{}`` is formatted once
        for each dataset name.

    Validation:
        Every requested set must be a key in ``SHA256``. Unknown names are
        reported together in sorted order. A URL not starting with ``http://``
        or ``https://`` is rejected.

    Processing:
        For each selected set, the destination is
        ``os.path.join(dir, "oxford", name)``. If ``has_images`` is True, only
        ``"<name>: already present in <dest>"`` is printed and no download
        occurs. Otherwise, ``download`` is called with
        ``args.url.format(name)``.

    Returns:
        This function returns normally after all selected sets are processed
        or skipped.

    Raises:
        SystemExit: With argparse's usage/error status when a set name is
            unknown or ``--url`` is not an HTTP or HTTPS URL.

    Examples:
        Supplying ``boat graf`` processes only those two sets. Supplying no set
        names processes bark, bikes, boat, graf, leuven, trees, ubc, and wall.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", help="test data folder (PopSift_TESTDATA_DIR)")
    parser.add_argument("sets", nargs="*", metavar="set",
                        help=", ".join(sorted(SHA256)))
    parser.add_argument("--url", default=URL,
                        help="archive URL with {} for the set name, e.g. a "
                             "mirror (default: the Oxford server)")
    args = parser.parse_args()
    unknown = set(args.sets) - set(SHA256)
    if unknown:
        parser.error(f"unknown set: {', '.join(sorted(unknown))}")
    if not args.url.startswith(("https://", "http://")):
        parser.error("--url must be an http or https URL")
    for name in args.sets or sorted(SHA256):
        dest = os.path.join(args.dir, "oxford", name)
        if has_images(dest):
            print(f"{name}: already present in {dest}")
        else:
            download(name, dest, args.url.format(name))


if __name__ == "__main__":
    main()
