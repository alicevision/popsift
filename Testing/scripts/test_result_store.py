#!/usr/bin/env python3
"""Unit tests of result_store.py: run names, feature files and stores.

    test_result_store.py [-v]

Needs only the Python standard library; CTest runs it as scripts/result_store.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import result_store as rs  # noqa: E402


def line(values):
    return " ".join(f"{v:g}" for v in values) + "\n"


KEYPOINT = [10.5, 20.25, 2.0, 1.0] + [float(i % 256) for i in range(128)]


class RunNames(unittest.TestCase):
    def test_configuration_alone(self):
        self.assertEqual(rs.run_name("popsift-vlfeat"), "popsift-vlfeat")

    def test_prefix(self):
        self.assertEqual(rs.run_name("popsift-vlfeat", "atan2-fast"),
                         "atan2-fast+popsift-vlfeat")

    def test_overrides_are_made_safe(self):
        self.assertEqual(rs.run_name("loop@gauss-mode=vlfeat", "x"),
                         "x+loop_gauss-mode=vlfeat")

    def test_bad_prefixes(self):
        for prefix in ("a+b", "a b", ".hidden", "a/b"):
            with self.assertRaises(rs.TestingError, msg=prefix):
                rs.run_name("vlfeat", prefix)


class ImageSets(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = self.dir.name
        for name in ("oxford/boat", "oxford/graf", "other"):
            os.makedirs(os.path.join(root, name))
        self.tools = {"image_sets": {
            label: os.path.join(root, label)
            for label in ("oxford/boat", "oxford/graf", "handcrafted")}}

    def tearDown(self):
        self.dir.cleanup()

    def expand(self, arg):
        with open(os.devnull, "w") as quiet:
            stdout, sys.stdout = sys.stdout, quiet
            try:
                return rs.expand_images(self.tools, arg)
            finally:
                sys.stdout = stdout

    def test_label(self):
        self.assertEqual(self.expand("oxford/boat"), ["oxford/boat"])

    def test_label_prefix(self):
        self.assertEqual(self.expand("oxford"), ["oxford/boat", "oxford/graf"])
        self.assertEqual(self.expand("oxford/"), ["oxford/boat",
                                                  "oxford/graf"])

    def test_prefix_ends_at_a_slash(self):
        with self.assertRaises(rs.TestingError):
            self.expand("oxf")

    def test_folder_of_a_configured_set_gets_its_label(self):
        folder = os.path.join(self.dir.name, "oxford", "boat")
        self.assertEqual(self.expand(folder), ["oxford/boat"])
        self.assertEqual(rs.resolve_images(self.tools, folder)[0],
                         "oxford/boat")

    def test_folder_above_configured_sets(self):
        self.assertEqual(self.expand(os.path.join(self.dir.name, "oxford")),
                         ["oxford/boat", "oxford/graf"])

    def test_other_folder_keeps_its_name(self):
        folder = os.path.join(self.dir.name, "other")
        self.assertEqual(self.expand(folder), [folder])
        self.assertEqual(rs.resolve_images(self.tools, folder)[0], "other")


class FeatureFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "f.sift")

    def tearDown(self):
        self.dir.cleanup()

    def write(self, text):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_vlfeat_format(self):
        self.write(line(KEYPOINT) + "\n" + line([1.0] + KEYPOINT[1:]))
        self.assertEqual(len(rs.read_features(self.path)), 2)

    def test_empty_file_has_no_keypoints(self):
        self.write("")
        self.assertEqual(rs.read_features(self.path), [])

    def test_oxford_format_is_rejected_by_name(self):
        self.write(line(KEYPOINT[:3] + [0.5, 0.0] + KEYPOINT[4:]))
        with self.assertRaisesRegex(rs.TestingError, "Oxford"):
            rs.read_features(self.path)

    def test_wrong_count(self):
        self.write(line(KEYPOINT[:-1]))
        with self.assertRaisesRegex(rs.TestingError, "131 numbers"):
            rs.read_features(self.path)

    def test_not_numbers(self):
        self.write("x " + line(KEYPOINT[1:]))
        with self.assertRaisesRegex(rs.TestingError, "not a line of numbers"):
            rs.read_features(self.path)

    def test_sorted_by_x(self):
        self.write(line([30.0] + KEYPOINT[1:]) + line(KEYPOINT))
        out = os.path.join(self.dir.name, "sorted.sift")
        rs.write_sorted(rs.read_features(self.path), out)
        with open(out, encoding="utf-8") as f:
            self.assertEqual([float(r.split()[0]) for r in f], [10.5, 30.0])


class Stores(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "atan2-fast+popsift-vlfeat")

    def tearDown(self):
        self.dir.cleanup()

    def manifest(self, **changes):
        m = {"format": rs.FORMAT, "name": "atan2-fast+popsift-vlfeat",
             "label": "fast atan2", "configuration": "popsift-vlfeat",
             "command": "popsift-demo -i {image}",
             "build": {"PopSift_TEST_WITH_VL_ATAN2": "ON"}}
        m.update(changes)
        return m

    def test_create_and_extend(self):
        rs.Store.create(self.path, self.manifest(), "oxford/boat")
        store = rs.Store.create(self.path, self.manifest(), "handcrafted")
        self.assertEqual(store.manifest["image_sets"],
                         ["handcrafted", "oxford/boat"])
        self.assertEqual(rs.Store.open(self.path).label, "fast atan2")

    def test_label_defaults_to_name(self):
        m = self.manifest()
        del m["label"]
        store = rs.Store.create(self.path, m, "handcrafted")
        self.assertEqual(store.label, store.name)

    def test_another_run_is_refused(self):
        rs.Store.create(self.path, self.manifest(), "handcrafted")
        with self.assertRaisesRegex(rs.TestingError, "another run"):
            rs.Store.create(self.path, self.manifest(
                build={"PopSift_TEST_WITH_VL_ATAN2": "OFF"}), "handcrafted")

    def test_another_run_replaces_with_replace(self):
        store = rs.Store.create(self.path, self.manifest(), "handcrafted")
        old = store.file("handcrafted", "box.pgm")
        os.makedirs(os.path.dirname(old))
        open(old, "w", encoding="utf-8").close()
        store = rs.Store.create(self.path, self.manifest(
            build={"PopSift_TEST_WITH_VL_ATAN2": "OFF"}), "oxford/boat",
            replace=True)
        self.assertFalse(os.path.exists(old))
        self.assertEqual(store.manifest["image_sets"], ["oxford/boat"])

    def test_files_follow_the_set_label(self):
        store = rs.Store.create(self.path, self.manifest(), "oxford/boat")
        self.assertEqual(store.file("oxford/boat", "img1.ppm"),
                         os.path.join(store.path, "oxford", "boat",
                                      "img1.sift"))
        self.assertFalse(store.has_set("oxford/graf"))

    def test_foreign_store(self):
        os.makedirs(self.path)
        with open(os.path.join(self.path, rs.MANIFEST), "w",
                  encoding="utf-8") as f:
            json.dump({"format": rs.FORMAT, "name": "other-tool"}, f)
        store = rs.find_store("atan2-fast+popsift-vlfeat", self.dir.name)
        self.assertEqual((store.name, store.label),
                         ("other-tool", "other-tool"))
        self.assertIsNone(rs.find_store("missing", self.dir.name))

    def test_manifest_needs_format_and_name(self):
        os.makedirs(self.path)
        for manifest in ({"name": "x"}, {"format": rs.FORMAT},
                         {"format": rs.FORMAT, "name": "a b"}):
            with open(os.path.join(self.path, rs.MANIFEST), "w",
                      encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(rs.TestingError, msg=str(manifest)):
                rs.Store.open(self.path)


if __name__ == "__main__":
    unittest.main()
