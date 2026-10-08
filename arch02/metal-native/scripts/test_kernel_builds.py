#!/usr/bin/env python3
"""Exercise real build.rs artifact contracts with a fake compiler, not GPU code.

The compiler fixture writes its output in place, as a linker may. Assertions
check held file descriptors, failed publication, dependency metadata, and
explicit prebuilt selection independently of shader math. All builds and files
are isolated in temporary directories.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METAL_NATIVE = ROOT / "arch_02_value_resid/metal-native"
GEMMA_METAL = ROOT / "gemma-metal"
CRATES = [METAL_NATIVE, GEMMA_METAL]
TESSL_KERNELS = ROOT / "crates/tessl/kernels"


class BuildPublication(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="metal-build-contract-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        fixture = (f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
name=Path(sys.argv[0]).name
if os.environ.get("FIXTURE_INVOCATIONS"):
    with Path(os.environ["FIXTURE_INVOCATIONS"]).open("a") as stream:
        stream.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
if name == "xcrun":
    print(Path(sys.argv[0]).parent/sys.argv[-1] if "-f" in sys.argv else "/tmp")
else:
    target=Path(sys.argv[sys.argv.index("-o")+1])
    target.write_bytes(os.environ.get("FIXTURE_CONTENT","new-library").encode())
    if name == "metallib" and os.environ.get("FIXTURE_FAIL_LINK") == "1":
        sys.exit(1)
''')
        for name in ["xcrun", "metal", "metallib"]:
            path = self.tools / name
            path.write_text(fixture)
            path.chmod(0o700)

    def setup_crate(self, source):
        root = self.root / source.name
        root.mkdir()
        shutil.copytree(source / "kernels", root / "kernels")
        tessl_kernels = root / "tessl-kernels"
        shutil.copytree(TESSL_KERNELS, tessl_kernels)
        out = root / "out"
        out.mkdir()
        binary = root / "build-script"
        subprocess.run(["rustc", "--edition=2021", str(source / "build.rs"), "-o", str(binary)],
                       check=True, capture_output=True, timeout=60)
        env = os.environ.copy()
        for key in [
            "METAL_NATIVE_SKIP_AOT",
            "GEMMA_METAL_SKIP_AOT",
            "GEMMA_METAL_PREBUILT_METALLIB",
            "FIXTURE_CONTENT",
            "FIXTURE_FAIL_LINK",
            "FIXTURE_INVOCATIONS",
        ]:
            env.pop(key, None)
        env.update(PATH=str(self.tools)+os.pathsep+env["PATH"], DEVELOPER_DIR="/tmp",
                   CARGO_MANIFEST_DIR=str(root), OUT_DIR=str(out),
                   DEP_TESSL_KERNELS=str(tessl_kernels))
        return root, binary, env

    def run_build(self, binary, env):
        return subprocess.run([str(binary)], env=env, capture_output=True, text=True, timeout=60)

    def test_rebuild_preserves_open_libraries(self):
        for source in CRATES:
            with self.subTest(crate=source.name):
                root, binary, env = self.setup_crate(source)
                first = self.run_build(binary, env)
                self.assertEqual(first.returncode, 0, first.stderr)
                path = Path(re.search(r"cargo:rustc-env=\w+_METALLIB=(.+)", first.stdout)[1])
                source_copy = root / "default.metallib"
                if source == GEMMA_METAL:
                    self.assertFalse(source_copy.exists(),
                                     "Gemma AOT must not mutate its source tree")
                with path.open("rb") as pinned:
                    offline_before = source_copy.read_bytes() if source_copy.exists() else None
                    env["FIXTURE_CONTENT"] = "different-next-library"
                    second = self.run_build(binary, env)
                    self.assertEqual(second.returncode, 0, second.stderr)
                    self.assertEqual(pinned.read(), b"new-library", "rebuild mutated a live library inode")
                    if offline_before is not None:
                        self.assertEqual(offline_before, b"new-library")
                    next_path = Path(re.search(r"cargo:rustc-env=\w+_METALLIB=(.+)", second.stdout)[1])
                    self.assertNotEqual(path, next_path, "builds must bake independent immutable paths")
                    self.assertEqual(next_path.read_bytes(), b"different-next-library")
                    if source == GEMMA_METAL:
                        self.assertFalse(source_copy.exists(),
                                         "Gemma rebuild must not mutate its source tree")

    def test_failed_link_does_not_publish(self):
        for source in CRATES:
            with self.subTest(crate=source.name):
                root, binary, env = self.setup_crate(source)
                first = self.run_build(binary, env)
                self.assertEqual(first.returncode, 0, first.stderr)
                path = Path(re.search(r"cargo:rustc-env=\w+_METALLIB=(.+)", first.stdout)[1])
                env.update(FIXTURE_FAIL_LINK="1", FIXTURE_CONTENT="partial-corrupt-library")
                failed = self.run_build(binary, env)
                self.assertNotEqual(failed.returncode, 0)
                self.assertEqual(path.read_bytes(), b"new-library", "failed linker corrupted prior output")
                source_copy = root / "default.metallib"
                if source == GEMMA_METAL:
                    self.assertFalse(source_copy.exists(),
                                     "failed Gemma build published into the source tree")
                else:
                    self.assertEqual(source_copy.read_bytes(), b"new-library")

    def test_offline_copy_failure_is_not_silenced(self):
        root, binary, env = self.setup_crate(METAL_NATIVE)
        (root/"default.metallib").mkdir()
        result = self.run_build(binary, env)
        self.assertNotEqual(result.returncode, 0, "offline artifact publication error was ignored")

    def test_gemma_skip_requires_explicit_prebuilt(self):
        _, binary, env = self.setup_crate(GEMMA_METAL)
        env["GEMMA_METAL_SKIP_AOT"] = "1"
        result = self.run_build(binary, env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GEMMA_METAL_PREBUILT_METALLIB", result.stderr)

    def test_gemma_skip_uses_exact_prebuilt_and_tracks_dependencies(self):
        root, binary, env = self.setup_crate(GEMMA_METAL)
        stale = root / "default.metallib"
        stale.write_bytes(b"stale-source-copy")
        artifact_dir = root / "artifacts"
        artifact_dir.mkdir()
        prebuilt = artifact_dir / "explicit-prebuilt.metallib"
        prebuilt.write_bytes(b"verified-prebuilt")
        configured = root / "prebuilt-alias.metallib"
        configured.symlink_to(prebuilt)
        env.update(GEMMA_METAL_SKIP_AOT="1",
                   GEMMA_METAL_PREBUILT_METALLIB=str(configured))

        result = self.run_build(binary, env)
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = Path(re.search(
            r"cargo:rustc-env=GEMMA_METAL_METALLIB=(.+)", result.stdout)[1])
        self.assertEqual(selected, prebuilt.resolve())
        self.assertEqual(selected.read_bytes(), b"verified-prebuilt")
        self.assertEqual(stale.read_bytes(), b"stale-source-copy")
        self.assertIn(f"cargo:rerun-if-changed={prebuilt.resolve()}", result.stdout)
        self.assertIn(
            "cargo:rerun-if-env-changed=GEMMA_METAL_PREBUILT_METALLIB",
            result.stdout,
        )
        self.assertIn("cargo:rerun-if-env-changed=DEP_TESSL_KERNELS", result.stdout)
        self.assertIn(
            f"cargo:rerun-if-changed="
            f"{(Path(env['DEP_TESSL_KERNELS']) / 'gelu.h').resolve()}",
            result.stdout,
        )

    def test_gemma_skip_rejects_relative_missing_and_non_file_prebuilts(self):
        root, binary, env = self.setup_crate(GEMMA_METAL)
        env["GEMMA_METAL_SKIP_AOT"] = "1"

        cases = [
            ("relative.metallib", "must be an absolute path"),
            (str(root / "missing.metallib"), "is not accessible"),
            (str(root), "is not a file"),
        ]
        for configured, diagnostic in cases:
            with self.subTest(configured=configured):
                env["GEMMA_METAL_PREBUILT_METALLIB"] = configured
                result = self.run_build(binary, env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_gemma_aot_resolves_tracks_and_passes_shared_header_directory(self):
        root, binary, env = self.setup_crate(GEMMA_METAL)
        invocations = root / "invocations.jsonl"
        configured_tessl_kernels = root / "tessl-kernels-alias"
        configured_tessl_kernels.symlink_to(Path(env["DEP_TESSL_KERNELS"]),
                                            target_is_directory=True)
        env["DEP_TESSL_KERNELS"] = str(configured_tessl_kernels)
        env["FIXTURE_INVOCATIONS"] = str(invocations)

        result = self.run_build(binary, env)
        self.assertEqual(result.returncode, 0, result.stderr)
        tessl_kernels = Path(env["DEP_TESSL_KERNELS"]).resolve()
        self.assertIn("cargo:rerun-if-env-changed=DEP_TESSL_KERNELS", result.stdout)
        self.assertIn(
            f"cargo:rerun-if-changed={tessl_kernels / 'gelu.h'}",
            result.stdout,
        )
        calls = [json.loads(line) for line in invocations.read_text().splitlines()]
        metal_calls = [call for call in calls if call[0] == "metal"]
        self.assertGreater(len(metal_calls), 0, "fixture observed no Metal compiler call")
        for call in metal_calls:
            include_index = call.index("-I")
            self.assertEqual(Path(call[include_index + 1]), tessl_kernels)

    def test_gemma_aot_requires_dependency_metadata_and_shared_header(self):
        root, binary, env = self.setup_crate(GEMMA_METAL)
        tessl_kernels = env.pop("DEP_TESSL_KERNELS")

        result = self.run_build(binary, env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DEP_TESSL_KERNELS not set", result.stderr)

        env["DEP_TESSL_KERNELS"] = tessl_kernels
        (Path(env["DEP_TESSL_KERNELS"]) / "gelu.h").unlink()

        result = self.run_build(binary, env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required Tessl shared GELU header missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
