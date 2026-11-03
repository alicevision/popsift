#!/usr/bin/env python3
"""
Downloads the Oxford affine-covariant region sets for the PopSift tests.

    python3 download_oxford.py <dir> [set ...]

Unpacks each set into <dir>/oxford/<set>/; without set names, all eight.
A set whose folder already holds images is skipped. Every archive is
checked against its SHA256 before it is unpacked.

The sets are published by the Visual Geometry Group, University of
Oxford, for research use, and are not redistributed with PopSift.
"""
import argparse
import hashlib
import io
import os
import sys
import tarfile
import urllib.request

URL = ("https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/{}.tar.gz")
SHA256 = {
    "bark": "f631c09a0f43fc1e092203f32f8f27e2e43e59b524918255d45474c50c91ea4f",
    "bikes": "3b95c49accadfd71e63c2fa0f7602abcbadd87d7f53ff54eb1efb15210d19e0c",
    "boat": "6e5721f37fbf9c3e974fd05935a42c1670f3e71aa3e5b218807ea046a068c603",
    "graf": "999871b945ee968a00a0d5f9af957d1382fb9dae1511cdee9553366817b53b5b",
    "leuven": "9b1ade6e64e5b27eee43463ff8f9ad2bd2f9c6911666e93de2634f3a5c0fe5dc",
    "trees": "04a39f201bffbc5035a560136fb37d2b36ecaa1532113eebfb113c90b5327f5d",
    "ubc": "26ab4ac94c8d0b2ded66c3bfaeb3f12589ec33005e5b4d5fcba4639f5b6e5414",
    "wall": "3cb3df75f40471f093adf6647d13df6a9f4312f7a222fc591d03f6c08e086344",
}


def has_images(folder):
    """Tell whether a folder exists and holds .pgm or .ppm files."""
    return os.path.isdir(folder) and any(
        f.lower().endswith((".pgm", ".ppm")) for f in os.listdir(folder))


def download(name, dest, url):
    """Download the archive of a set, check it and unpack it into dest."""
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
                    f.write(tar.extractfile(member).read())
    print(f"{name}: unpacked into {dest}")


def main():
    """Download the sets named on the command line, or all of them."""

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
