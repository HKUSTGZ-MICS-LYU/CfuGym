"""Offline contracts for trusted Embench patches and the split Huffbench oracle.

Only lightweight native compilation is used: no make, simulator, synthesis, model,
network, or persistent run workspace is involved.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import yaml

from agents.cfu_design_agent.benchmark import embench, embench_integrity as integrity
from agents.cfu_design_agent.benchmark import embench_patchers as patchers
from agents.cfu_design_agent.runtime.policy import load_policy
from agents.cfu_design_agent.runtime.tools import RepoToolbox
from agents.tests.common import REPO_ROOT
from agents.tests.test_embench import temporary_embench_repo


def _sources(name: str) -> dict[str, str]:
    return {Path(rel).name: (REPO_ROOT / rel).read_text()
            for rel in embench.benchmark_source_files(REPO_ROOT, name)}


def _replace_body(source: str, name: str, body: str) -> str:
    parsed = integrity._parse(source)
    function = next(obj for obj in parsed.objects
                    if obj.name == name and obj.body is not None)
    brace = next(token for token in parsed.tokens
                 if function.start <= token.start < function.end and token.value == "{")
    return source[:brace.start] + "{\n" + body + "\n}" + source[function.end:]


class PatchPipelineTest(unittest.TestCase):
    def test_deterministic_results_and_caller_sources_not_mutated(self) -> None:
        for name in ("crc32", "statemate", "huffbench"):
            with self.subTest(name=name):
                sources = _sources(name)
                original = copy.deepcopy(sources)
                first = patchers.apply_benchmark_patches(name, sources)
                second = patchers.apply_benchmark_patches(name, dict(reversed(list(sources.items()))))
                self.assertEqual(sources, original)
                self.assertIsNot(first.sources, sources)
                for attribute in ("sources", "support_files", "patches", "notes"):
                    self.assertEqual(getattr(first, attribute), getattr(second, attribute))
                self.assertEqual(first.patches, patchers.patch_manifest(name))
                self.assertTrue(first.patches)
                ids = [patch["id"] for patch in first.patches]
                expected_ids = {"local_scale_factor"}
                if name == "statemate":
                    expected_ids.add("statemate_link_symbol")
                if name == "huffbench":
                    expected_ids.add("huffbench_stages")
                    staged = next(patch for patch in first.patches if patch["id"] == "huffbench_stages")
                    self.assertRegex(staged["assets_sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(set(ids), expected_ids)
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(patch["version"] == 1 for patch in first.patches))
                # No global manifest/result state may leak through a caller edit.
                first.patches[0]["id"] = "caller-corrupted"
                self.assertEqual(second.patches, patchers.patch_manifest(name))

    def test_scale_normalization_and_statemate_legacy_api(self) -> None:
        for name in ("crc32", "statemate", "huffbench"):
            with self.subTest(name=name):
                result = patchers.apply_benchmark_patches(name, _sources(name))
                definitions = [token.value
                    for text in [*result.sources.values(), *result.support_files.values()]
                    for token in integrity.scan(text)
                    if token.directive and token.value.split()[:3] == ["#", "define", "LOCAL_SCALE_FACTOR"]]
                self.assertEqual(definitions, ["# define LOCAL_SCALE_FACTOR 1"])
        sources = _sources("statemate")
        result = patchers.apply_benchmark_patches("statemate", sources)
        scaled = integrity.normalize_local_scale_factor(sources)
        normalized, notes = embench.normalize_embench_link_symbols("statemate", scaled)
        self.assertEqual(normalized, result.sources)
        self.assertEqual(notes, result.notes)
        tokens = [token.value for text in normalized.values()
                  for token in integrity.scan(text) if not token.directive]
        self.assertNotIn("time", tokens)
        self.assertIn("statemate_timestamp", tokens)
        self.assertEqual(sources, _sources("statemate"))

    def test_ambiguous_scale_definitions_fail_closed(self) -> None:
        cases = ({"a.c": "int a;\n"},
                 {"a.c": "#define LOCAL_SCALE_FACTOR 7\n", "b.c": "#define LOCAL_SCALE_FACTOR 9\n"},
                 {"a.c": "#define LOCAL_SCALE_FACTOR 7\n#undef LOCAL_SCALE_FACTOR\n"})
        for sources in cases:
            with self.subTest(sources=sources):
                original = dict(sources)
                with self.assertRaises(ValueError):
                    patchers.apply_benchmark_patches("fixture", sources)
                self.assertEqual(sources, original)

    def test_upstream_source_drift_fails_closed(self) -> None:
        cases = []
        statemate = _sources("statemate")
        changed = re.sub(r"\bunsigned\s+long\s+time\s*;", "unsigned int time;",
                         statemate["libstatemate.c"], count=1)
        self.assertNotEqual(changed, statemate["libstatemate.c"])
        cases.append(("statemate", {**statemate, "libstatemate.c": changed}))
        huffbench = _sources("huffbench")
        for before, after in (("compdecomp", "upstream_compdecomp"),
                              ("#define TEST_SIZE 500", "#define TEST_SIZE 499"),
                              ("'J', '2', 'O'", "'X', '2', 'O'")):
            changed = huffbench["libhuffbench.c"].replace(before, after, 1)
            self.assertNotEqual(changed, huffbench["libhuffbench.c"])
            cases.append(("huffbench", {**huffbench, "libhuffbench.c": changed}))
        cases.append(("huffbench", {"renamed.c": huffbench["libhuffbench.c"]}))
        for name, sources in cases:
            with self.subTest(name=name, filenames=list(sources)):
                original = dict(sources)
                with self.assertRaises(ValueError):
                    patchers.apply_benchmark_patches(name, sources)
                self.assertEqual(sources, original)

    def test_huffbench_separates_private_fixture_and_entrypoints(self) -> None:
        result = patchers.apply_benchmark_patches("huffbench", _sources("huffbench"))
        self.assertEqual(set(result.sources), {"libhuffbench.c"})
        assets = {Path(name).name: text for name, text in result.support_files.items()}
        self.assertEqual(set(assets), {"huffbench.h", "huffbench_harness.c"})
        kernel = integrity._parse(result.sources["libhuffbench.c"])
        functions = {obj.name for obj in kernel.objects if obj.body is not None}
        self.assertTrue({"huffbench_encode", "huffbench_decode"} <= functions)
        self.assertFalse(integrity.ENTRY_FUNCTIONS & functions)
        names = set().union(*(integrity._declared_names(obj) for obj in kernel.objects))
        self.assertFalse({"orig_data", "test_data", "huffbench_decoded", "huffbench_input"} & names)
        harness = integrity._parse(assets["huffbench_harness.c"])
        harness_functions = {obj.name for obj in harness.objects if obj.body is not None}
        self.assertTrue(integrity.ENTRY_FUNCTIONS <= harness_functions)
        harness_names = set().union(*(integrity._declared_names(obj) for obj in harness.objects))
        self.assertIn("orig_data", harness_names)
        self.assertTrue({"test_data", "huffbench_decoded"} & harness_names)


class PreparedPatchIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workdir = temporary_embench_repo(self)
        self.run_root = "agents/generated/runs/patchers-test"
        self.name = "huffbench"
        self.original = {name: text for name, text in _sources(self.name).items()}
        self.patched = patchers.apply_benchmark_patches(self.name, self.original)
        self.project = embench.prepare_embench_benchmark(RepoToolbox(
            self.workdir, policy=load_policy(REPO_ROOT), stage="prepare_benchmark",
            run_root=self.run_root), {"benchmark_name": self.name})
        self.metadata = self.workdir / self.run_root / "04_benchmark/embench.yaml"
        self.metadata.parent.mkdir(parents=True, exist_ok=True)
        self.write_metadata(self.project)
        self.target = f"{embench.config_workspace(self.run_root, self.name, 'cfu')}/src/libhuffbench.c"

    def write_metadata(self, project: dict) -> None:
        self.metadata.write_text(yaml.safe_dump(project))

    def check(self, pending: dict[str, str] | None = None) -> list[str]:
        return integrity.validate_candidate(self.workdir, self.run_root, self.name, pending)

    def asset_paths(self) -> list[str]:
        return [f"{embench.config_workspace(self.run_root, self.name, config)}/support/{filename}"
                for config in embench.CONFIGS
                for filename in ("huffbench.h", "huffbench_harness.c")]

    def test_all_configs_have_equal_patched_kernels_assets_and_metadata(self) -> None:
        self.assertEqual(self.project["patches"], self.patched.patches)
        self.assertEqual(self.project["patches"], patchers.patch_manifest(self.name))
        self.assertEqual(self.project["integrity"]["version"], integrity.INTEGRITY_VERSION)
        self.assertEqual(self.project["verification"], "bench_verify")
        refs = self.project["integrity"]["reference_sources"]
        protected = self.project["integrity"]["protected_files"]
        for config in embench.CONFIGS:
            with self.subTest(config=config):
                cfg = embench.config_workspace(self.run_root, self.name, config)
                for filename, text in self.patched.sources.items():
                    rel = f"{cfg}/src/{filename}"
                    self.assertEqual((self.workdir / rel).read_text(), text)
                    if config == "reference":
                        self.assertEqual(refs[rel], hashlib.sha256(text.encode()).hexdigest())
                for filename, text in self.patched.support_files.items():
                    rel = f"{cfg}/support/{Path(filename).name}"
                    self.assertEqual((self.workdir / rel).read_text(), text)
                    self.assertIn(rel, self.project["copied_files"])
                    self.assertEqual(protected[rel], hashlib.sha256(text.encode()).hexdigest())
                command = embench.build_command(self.run_root, self.name, config)
                extra = next(arg for arg in command if arg.startswith("EXTRA_SOURCES="))
                self.assertIn("/support/huffbench_harness.c", extra)
        self.assertEqual(self.check(), [])
        upstream = {Path(rel).name: (self.workdir / rel).read_text()
                    for rel in embench.benchmark_source_files(self.workdir, self.name)}
        self.assertEqual(upstream, self.original)

    def test_protected_header_and_harness_pending_edits_rejected_in_every_config(self) -> None:
        self.assertEqual(self.check(), [])
        for rel in self.asset_paths():
            with self.subTest(path=rel):
                text = (self.workdir / rel).read_text()
                self.assertEqual(self.check({rel: text}), [])
                self.assertTrue(self.check({rel: text + "\n/* changed protected bytes */\n"}))
                self.assertTrue(self.check({rel: ""}))
                self.assertEqual((self.workdir / rel).read_text(), text)

    def test_protected_header_and_harness_disk_edits_and_deletion_rejected(self) -> None:
        for rel in self.asset_paths():
            with self.subTest(path=rel):
                path = self.workdir / rel
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"\n/* tampered */\n")
                    self.assertTrue(self.check())
                    path.unlink()
                    self.assertTrue(self.check())
                finally:
                    path.write_bytes(original)
        self.assertEqual(self.check(), [])

    def test_missing_legacy_or_incomplete_patch_metadata_rejected(self) -> None:
        missing = copy.deepcopy(self.project)
        missing.pop("patches")
        cases = [missing, {**self.project, "patches": []},
                 {**self.project, "patches": [{"id": "legacy_huffbench_roundtrip"}]}]
        for index in range(len(self.project["patches"])):
            project = copy.deepcopy(self.project)
            project["patches"].pop(index)
            cases.append(project)
            legacy = copy.deepcopy(self.project)
            legacy["patches"][index]["version"] = 0
            cases.append(legacy)
        stale_assets = copy.deepcopy(self.project)
        staged = next(patch for patch in stale_assets["patches"] if patch["id"] == "huffbench_stages")
        staged["assets_sha256"] = "0" * 64
        cases.append(stale_assets)
        for project in cases:
            with self.subTest(patches=project.get("patches")):
                self.write_metadata(project)
                self.assertTrue(self.check())
        self.write_metadata(self.project)
        self.assertEqual(self.check(), [])

    def test_omitting_required_protected_asset_hash_does_not_disable_protection(self) -> None:
        for rel in self.asset_paths():
            with self.subTest(path=rel):
                project = copy.deepcopy(self.project)
                project["integrity"]["protected_files"].pop(rel)
                self.write_metadata(project)
                self.assertTrue(self.check())
        self.write_metadata(self.project)

    def test_codec_body_edits_allowed_without_changing_trusted_files(self) -> None:
        original = (self.workdir / self.target).read_text()
        for name in ("huffbench_encode", "huffbench_decode"):
            with self.subTest(function=name):
                changed = _replace_body(original, name, "/* computational edit, not a correctness claim */\nreturn 1;")
                self.assertNotEqual(changed, original)
                self.assertEqual(self.check({self.target: changed}), [])
        self.assertEqual((self.workdir / self.target).read_text(), original)

    def test_memory_routine_interposition_rejected(self) -> None:
        original = (self.workdir / self.target).read_text()
        replacements = {
            "memset": "void *memset(void *destination, int value, size_t count) { return destination; }",
            "memcpy": "void *memcpy(void *destination, const void *source, size_t count) { return destination; }",
            "memcmp": "int memcmp(const void *left, const void *right, size_t count) { return 0; }",
        }
        for name, function in replacements.items():
            with self.subTest(function=name):
                self.assertTrue(self.check({self.target: original + "\n" + function + "\n"}))
        self.assertEqual((self.workdir / self.target).read_text(), original)

    def test_private_helper_redefinitions_and_authority_macro_aliases_rejected(self) -> None:
        original = (self.workdir / self.target).read_text()
        attacks = [
            "static int huffbench_reference_encode(const unsigned char *data, size_t count, struct huffbench_encoded *encoded) { return 1; }",
            "static void huffbench_reference_heap_adjust(size_t *frequency, size_t *heap, int count, int key) {}",
            "#define huffbench_reference_encode(...) 1",
            "#define huffbench_reference cfu_shadow",
            "#define CFU_INIT initialise_benchmark",
            "#define CFU_TRIGGER start_trigger",
        ]
        for attack in attacks:
            with self.subTest(attack=attack):
                self.assertTrue(self.check({self.target: original + "\n" + attack + "\n"}))
        self.assertEqual((self.workdir / self.target).read_text(), original)


_NATIVE_MAIN = '''#include <stdio.h>
#include <stdint.h>
#include "support.h"
#include "huffbench.h"
_Static_assert(HUFFBENCH_TEST_SIZE == 500, "fixture size");
_Static_assert(HUFFBENCH_CAPACITY == 501, "encoded capacity");
_Static_assert(sizeof(((struct huffbench_encoded *)0)->code[0]) == 4, "32-bit code");
_Static_assert(sizeof(((struct huffbench_encoded *)0)->code) == 256 * sizeof(uint32_t), "code capacity");
_Static_assert(sizeof(((struct huffbench_encoded *)0)->clen) == 256, "code lengths");
_Static_assert(sizeof(((struct huffbench_encoded *)0)->bytes) == 501, "byte capacity");
int main(void) {
    int (*encode)(const unsigned char *, size_t, struct huffbench_encoded *) = huffbench_encode;
    int (*decode)(const struct huffbench_encoded *, unsigned char *, size_t) = huffbench_decode;
    (void)encode; (void)decode;
    initialise_benchmark();
    int first = verify_benchmark(benchmark());
    int second = verify_benchmark(benchmark());
    printf("%d %d\\n", first, second);
    return (first == 1 && second == 1) ? 0 : 1;
}
'''


class NativeHuffbenchOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = patchers.apply_benchmark_patches("huffbench", _sources("huffbench"))

    def compile_command(self, compiler: str, root: Path, kernel: str,
                        *, syntax_only: bool = False) -> list[str]:
        support = root / "support"
        support.mkdir()
        for name in ("support.h", "beebsc.h", "beebsc.c"):
            shutil.copyfile(REPO_ROOT / embench.EMBENCH_ROOT / "support" / name, support / name)
        for name, text in self.result.support_files.items():
            (support / Path(name).name).write_text(text)
        sources = []
        for name, text in self.result.sources.items():
            path = root / Path(name).name
            path.write_text(kernel if path.name == "libhuffbench.c" else text)
            sources.append(str(path))
        (root / "main.c").write_text(_NATIVE_MAIN)
        command = [compiler, "-std=gnu11", "-DGLOBAL_SCALE_FACTOR=1", "-I" + str(support),
                   "-Werror=incompatible-pointer-types", *sources,
                   str(support / "huffbench_harness.c"), str(support / "beebsc.c"), str(root / "main.c")]
        if syntax_only:
            command.extend(["-fsyntax-only", "-march=rv32imc_zicsr_zifencei", "-mabi=ilp32"])
        else:
            command.extend(["-O1", "-o", str(root / "benchmark")])
        return command

    def execute(self, kernel: str, *, accepted: bool) -> None:
        compiler = shutil.which("gcc")
        if not compiler:
            self.skipTest("native GCC not installed")
        with tempfile.TemporaryDirectory(prefix="embench-patchers-") as directory:
            root = Path(directory)
            built = subprocess.run(self.compile_command(compiler, root, kernel),
                                   capture_output=True, text=True, timeout=15, check=False)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(root / "benchmark")], capture_output=True,
                                 text=True, timeout=5, check=False)
            self.assertEqual(run.returncode, 0 if accepted else 1, run.stdout + run.stderr)
            self.assertEqual(run.stdout.strip(), "1 1" if accepted else "0 0")

    def test_original_patched_codec_passes_native_benchmark_twice(self) -> None:
        self.execute(self.result.sources["libhuffbench.c"], accepted=True)

    def test_different_optimal_prefix_code_and_stream_are_accepted(self) -> None:
        kernel = re.sub(r"\bhuffbench_encode\b", "honest_encode",
                        self.result.sources["libhuffbench.c"])
        kernel += '''\nint huffbench_encode(const unsigned char *data, size_t data_len,
                      struct huffbench_encoded *encoded) {
    int result = honest_encode(data, data_len, encoded);
    if (!result) return result;
    size_t bits = 0;
    for (size_t index = 0; index < data_len; ++index) bits += encoded->clen[data[index]];
    for (size_t symbol = 0; symbol < 256; ++symbol) {
        unsigned length = encoded->clen[symbol];
        if (length) {
            uint32_t mask = length == 32 ? UINT32_MAX : (UINT32_C(1) << length) - 1;
            encoded->code[symbol] ^= mask;
        }
    }
    /* Complement only actual codeword bits, never the zero padding. */
    for (size_t bit = 0; bit < bits; ++bit)
        encoded->bytes[bit / 8] ^= (unsigned char)(0x80u >> (bit % 8));
    return result;
}
'''
        self.execute(kernel, accepted=True)

    def test_both_noop_functions_returning_success_fail(self) -> None:
        kernel = self.result.sources["libhuffbench.c"]
        for name in ("huffbench_encode", "huffbench_decode"):
            kernel = _replace_body(kernel, name, "return 1;")
        self.execute(kernel, accepted=False)

    def test_honest_encode_with_noop_decode_fails(self) -> None:
        kernel = _replace_body(self.result.sources["libhuffbench.c"], "huffbench_decode", "return 1;")
        self.execute(kernel, accepted=False)

    def test_honest_encode_with_corrupted_bitstream_fails(self) -> None:
        kernel = self.result.sources["libhuffbench.c"]
        # Keep the actual encoder intact; corrupt only its emitted stream.
        kernel = re.sub(r"\bhuffbench_encode\b", "honest_encode", kernel)
        kernel += '''\nint huffbench_encode(const unsigned char *data, size_t data_len,
                      struct huffbench_encoded *encoded) {
    int result = honest_encode(data, data_len, encoded);
    encoded->bytes[0] ^= 0x80u;
    return result;
}
'''
        self.execute(kernel, accepted=False)

    def test_corrupting_only_last_decoded_byte_fails(self) -> None:
        kernel = re.sub(r"\bhuffbench_decode\b", "honest_decode",
                        self.result.sources["libhuffbench.c"])
        kernel += '''\nint huffbench_decode(const struct huffbench_encoded *encoded,
                      unsigned char *data, size_t data_len) {
    int result = honest_decode(encoded, data, data_len);
    if (data_len) data[data_len - 1] ^= 1u;
    return result;
}
'''
        self.execute(kernel, accepted=False)

    def test_malformed_encoded_records_and_failed_encoder_stage_are_rejected(self) -> None:
        mutations = {
            "zero-length": "encoded->length = 0;",
            "over-capacity": "encoded->length = HUFFBENCH_CAPACITY + 1;",
            "oversized-code": "encoded->clen[data[0]] = 33;",
            "duplicate-code": '''
                encoded->code[data[1]] = encoded->code[data[0]];
                encoded->clen[data[1]] = encoded->clen[data[0]];
''',
            "proper-prefix-collision": '''
                encoded->code[data[1]] = encoded->code[data[0]] << 1;
                encoded->clen[data[1]] = encoded->clen[data[0]] + 1;
''',
            "failed-encoder-stage": "result = 0;",
            "nonzero-final-padding": '''
                size_t bits = 0;
                for (size_t index = 0; index < data_len; ++index)
                    bits += encoded->clen[data[index]];
                /* Fail loudly if a future fixture has no padding to test. */
                if (bits % 8 == 0) __builtin_trap();
                encoded->bytes[encoded->length - 1] |= 1u;
''',
        }
        original = re.sub(r"\bhuffbench_encode\b", "honest_encode",
                          self.result.sources["libhuffbench.c"])
        for label, mutation in mutations.items():
            with self.subTest(mutation=label):
                kernel = original + '''\nint huffbench_encode(const unsigned char *data, size_t data_len,
                      struct huffbench_encoded *encoded) {
    int result = honest_encode(data, data_len, encoded);
    if (!result) return result;
''' + mutation + '''
    return result;
}
'''
                self.execute(kernel, accepted=False)

    def test_fixed_eight_bit_identity_codec_fails_optimality(self) -> None:
        kernel = self.result.sources["libhuffbench.c"]
        kernel = _replace_body(kernel, "huffbench_encode", '''
    if (data_len > HUFFBENCH_CAPACITY) return 0;
    for (size_t symbol = 0; symbol < 256; ++symbol) {
        encoded->code[symbol] = (uint32_t)symbol;
        encoded->clen[symbol] = 0;
    }
    encoded->length = data_len;
    for (size_t index = 0; index < data_len; ++index) {
        encoded->clen[data[index]] = 8;
        encoded->bytes[index] = data[index];
    }
    return 1;
''')
        # With the honest decoder still processing the protected reference
        # stream, rejection must come from the encoder's nonoptimal output.
        with self.subTest(decoder="honest"):
            self.execute(kernel, accepted=False)
        kernel = _replace_body(kernel, "huffbench_decode", '''
    for (size_t index = 0; index < data_len; ++index) data[index] = encoded->bytes[index];
    return 1;
''')
        with self.subTest(decoder="identity"):
            self.execute(kernel, accepted=False)

    def test_riscv_cross_compiler_syntax_if_available(self) -> None:
        compiler = next((path for name in ("riscv64-unknown-elf-gcc", "riscv32-unknown-elf-gcc")
                         if (path := shutil.which(name))), None)
        if not compiler:
            self.skipTest("RISC-V GCC not installed")
        with tempfile.TemporaryDirectory(prefix="embench-patchers-syntax-") as directory:
            command = self.compile_command(compiler, Path(directory),
                self.result.sources["libhuffbench.c"], syntax_only=True)
            result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
