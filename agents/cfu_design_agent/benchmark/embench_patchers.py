"""Deterministic, Host-owned patches of isolated Embench source copies.

Patches run once, before any config is copied/hashed. Generated support files
are immutable harness inputs, not accelerator implementation targets. Patch
identity is checked on resume as well as in new reference/profile cache keys.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from . import embench_integrity, embench_instrument


@dataclass
class BenchmarkPatchResult:
    sources: dict[str, str]
    support_files: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    patches: list[dict] = field(default_factory=list)


class BenchmarkPatcher(ABC):
    """One explicit, versioned source transformation; no build/flow side effects."""

    patch_id: str
    version = 1
    benchmark: str | None = None
    support_names: tuple[str, ...] = ()

    def applies_to(self, name: str) -> bool:
        return self.benchmark is None or name == self.benchmark

    def identity(self) -> dict:
        digest = hashlib.sha256()
        for filename in (__file__, embench_integrity.__file__, embench_instrument.__file__):
            path = Path(filename)
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return {"id": self.patch_id, "version": self.version,
                "implementation_sha256": digest.hexdigest()}

    @abstractmethod
    def apply(self, sources: dict[str, str]) -> BenchmarkPatchResult:
        """Return fresh source maps; refuse unfamiliar source instead of guessing."""


class LocalScaleFactorPatcher(BenchmarkPatcher):
    patch_id = "local_scale_factor"

    def apply(self, sources: dict[str, str]) -> BenchmarkPatchResult:
        return BenchmarkPatchResult(embench_integrity.normalize_local_scale_factor(sources))


class StatemateSymbolPatcher(BenchmarkPatcher):
    patch_id = "statemate_link_symbol"
    benchmark = "statemate"

    def apply(self, sources: dict[str, str]) -> BenchmarkPatchResult:
        filename = "libstatemate.c"
        text = sources.get(filename)
        if text is None:
            raise ValueError("statemate compatibility rename requires libstatemate.c")
        tokens = [token for token in embench_integrity.scan(text) if not token.directive]
        values = [token.value for token in tokens]
        declaration = [i for i in range(len(values) - 3)
                       if values[i:i + 4] == ["unsigned", "long", "time", ";"]]
        if len(declaration) != 1:
            raise ValueError("statemate time-symbol collision changed; expected one 'unsigned long time;' declaration")
        time_tokens = [i for i, token in enumerate(tokens) if token.value == "time"]
        for i in time_tokens:
            if ((i + 1 < len(tokens) and tokens[i + 1].value == "(")
                    or (i > 0 and tokens[i - 1].value in {".", "->"})):
                raise ValueError("statemate time-symbol rename found a function/member use; review compatibility fix")
        updated = text
        for i in reversed(time_tokens):
            token = tokens[i]
            updated = updated[:token.start] + "statemate_timestamp" + updated[token.end:]
        result = dict(sources)
        result[filename] = updated
        return BenchmarkPatchResult(result, notes=[
            "statemate build compatibility: renamed its global time variable to "
            "statemate_timestamp in isolated copies to avoid the Vexii time() link collision"
        ])


_ASSETS = Path(__file__).with_name("patches")
_HUFFBENCH_SOURCE_SHA256 = "52ae5e0b35653939ae4f103ceed76eebdfafde4ee330c4ce9b98d261a42f3ba5"


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("huffbench source anchor changed: " + old[:80])
    return text.replace(old, new, 1)


def _definition(text: str, name: str) -> str:
    matches = [obj for obj in embench_integrity._parse(text).objects
               if obj.name == name and obj.body is not None]
    if len(matches) != 1:
        raise ValueError(f"huffbench requires one {name} definition")
    obj = matches[0]
    return text[obj.start:obj.end]


class HuffbenchPatcher(BenchmarkPatcher):
    patch_id = "huffbench_stages"
    benchmark = "huffbench"
    support_names = ("huffbench.h", "huffbench_harness.c")

    def identity(self) -> dict:
        digest = hashlib.sha256()
        for name in self.support_names:
            digest.update(name.encode())
            digest.update((_ASSETS / name).read_bytes())
        return {**super().identity(), "assets_sha256": digest.hexdigest()}

    def apply(self, sources: dict[str, str]) -> BenchmarkPatchResult:
        text = sources.get("libhuffbench.c", "")
        if hashlib.sha256(text.encode()).hexdigest() != _HUFFBENCH_SOURCE_SHA256:
            raise ValueError("huffbench source revision changed; review the split-stage patch before preparing a new run")
        heap = _definition(text, "heap_adjust")
        # Preserve the 1-based heap algorithm without forming an out-of-array
        # pointer. Both the DUT seed and trusted reference get this UB fix.
        for old, new in (
            ("  --heap;", "  /* Array accesses below translate 1-based indices. */"),
            ("int v = heap[k];", "int v = heap[k - 1];"),
            ("freq[heap[j]] > freq[heap[j + 1]]", "freq[heap[j - 1]] > freq[heap[j]]"),
            ("if (freq[v] < freq[heap[j]])", "if (freq[v] < freq[heap[j - 1]])"),
            ("heap[k] = heap[j];", "heap[k - 1] = heap[j - 1];"),
            ("heap[k] = v;", "heap[k - 1] = v;"),
        ):
            heap = _replace_once(heap, old, new)
        original = _definition(text, "compdecomp")
        body = original[embench_instrument.find_definition_brace(original, "compdecomp") + 1:original.rfind("}")]
        marker = "  /*\n     DECOMPRESSION\n   */"
        if body.count(marker) != 1:
            raise ValueError("huffbench compression/decompression boundary changed")
        compression, decompression = body.split(marker)
        compression = _replace_once(compression, "  byte *cptr;\n", "")
        compression = _replace_once(compression, "  char bout = 0;", "  byte bout = 0;")
        compression = _replace_once(compression, "  byte *dptr = data;", "  const byte *dptr = data;")
        compression = _replace_once(compression,
            "  byte *comp = (byte *) malloc_beebs (data_len + 1);",
            "  byte *comp = encoded->bytes;")
        compression = _replace_once(compression, "  bits32 code[256];", "  bits32 *code = encoded->code;")
        compression = _replace_once(compression, "  byte clen[256];", "  byte *clen = encoded->clen;")
        compression = compression.replace("return;", "return 0;")
        compression = _replace_once(compression, "(sizeof (unsigned long) * 8)", "32")
        compression = _replace_once(compression, "mask = 1 << (clen[(*dptr)] - 1);",
                                     "mask = UINT32_C(1) << (clen[(*dptr)] - 1);")
        encoder = ("int\nhuffbench_encode(const byte *data, size_t data_len,\n"
                   "                 struct huffbench_encoded *encoded)\n{\n"
                   "  if (!data_len || data_len > HUFFBENCH_TEST_SIZE) return 0;\n"
                   "  encoded->length = 0;\n" + compression +
                   "\n  encoded->length = comp_len;\n  return 1;\n}\n")
        decompression = _replace_once(decompression,
            "  // remove work areas\n  free_beebs (comp);", "  return 1;")
        decompression = _replace_once(decompression,
            "  for (j = 0; heap2[j] == 0; ++j);",
            "  for (j = 0; j < 256 && heap2[j] == 0; ++j);\n  if (j == 256) return 0;")
        decompression = _replace_once(decompression, "  cptr = comp;", "  cptr = encoded->bytes;")
        decompression = _replace_once(decompression, "  while (n < data_len)\n    {",
            "  while (n < data_len)\n    {\n      if (cptr >= encoded->bytes + encoded->length) return 0;")
        decompression = _replace_once(decompression, "      while (heap2[i] < k)",
            "      while (i < 256 && heap2[i] < k)")
        decompression = _replace_once(decompression, "      // code matches, character found",
            "      if (i == 256) return 0;\n\n      // code matches, character found")
        decompression = _replace_once(decompression, "mask = 1 << (clen[j] - 1);",
            "mask = UINT32_C(1) << (clen[j] - 1);")
        decoder = ("int\nhuffbench_decode(const struct huffbench_encoded *encoded,\n"
                   "                 byte *data, size_t data_len)\n{\n"
                   "  size_t i, j, n, mask;\n  bits32 k, t;\n  byte c;\n"
                   "  const byte *cptr;\n  byte *dptr;\n"
                   "  const bits32 *code = encoded->code;\n"
                   "  const byte *clen = encoded->clen;\n"
                   "  if (!encoded->length || encoded->length > HUFFBENCH_CAPACITY ||\n"
                   "      !data_len || data_len > HUFFBENCH_TEST_SIZE) return 0;\n"
                   "  for (i = 0; i < 256; ++i)\n"
                   "    if (clen[i] > 32 || (!clen[i] && code[i]) ||\n"
                   "        (clen[i] && clen[i] < 32 && (code[i] >> clen[i]))) return 0;\n" +
                   decompression + "\n}\n")
        license_text = text[:text.index("#include <string.h>")]
        kernel = (license_text + '/* Split-stage computational bodies; driver/oracle are Host-owned. */\n'
                  '#include "huffbench.h"\n#include <string.h>\n'
                  'typedef uint32_t bits32;\ntypedef unsigned char byte;\n\n' +
                  heap + "\n\n" + encoder + "\n" + decoder)
        start = text.index("static const byte orig_data[TEST_SIZE]")
        end = text.index(";", start) + 1
        fixture = text[start:end].replace("[TEST_SIZE]", "[HUFFBENCH_TEST_SIZE]")
        reference = (heap.replace("heap_adjust", "huffbench_reference_heap_adjust") + "\n\nstatic " +
                     encoder.replace("huffbench_encode(", "huffbench_reference_encode(")
                     .replace("heap_adjust", "huffbench_reference_heap_adjust"))
        harness = (_ASSETS / "huffbench_harness.c").read_text()
        for marker, replacement in (("@LICENSE@", license_text), ("@INPUT@", fixture),
                                    ("@REFERENCE_CODEC@", reference)):
            harness = _replace_once(harness, marker, replacement)
        result = dict(sources)
        result["libhuffbench.c"] = kernel
        return BenchmarkPatchResult(result, support_files={
            "huffbench.h": (_ASSETS / "huffbench.h").read_text(),
            "huffbench_harness.c": harness,
        }, notes=[
            "huffbench_stages_v1: timed encode/decode APIs with Host-owned driver; "
            "decoder receives an independent reference stream, output is poisoned, "
            "and the oracle checks every encoded bit, prefix-free optimal codebook "
            "and all decoded bytes. Original fixed 500-byte fixture only; not hidden-input validation. "
            "New baseline includes stage reset/copy overhead and is not comparable with old roundtrip timings."
        ])


PATCHERS: tuple[BenchmarkPatcher, ...] = (
    LocalScaleFactorPatcher(), StatemateSymbolPatcher(), HuffbenchPatcher(),
)


def patch_manifest(name: str) -> list[dict]:
    return [patcher.identity() for patcher in PATCHERS if patcher.applies_to(name)]


def extra_support_files(name: str) -> tuple[str, ...]:
    return tuple(filename for patcher in PATCHERS if patcher.applies_to(name)
                 for filename in patcher.support_names)


def apply_benchmark_patches(name: str, sources: dict[str, str]) -> BenchmarkPatchResult:
    result = BenchmarkPatchResult(dict(sources))
    for patcher in PATCHERS:
        if not patcher.applies_to(name):
            continue
        patched = patcher.apply(dict(result.sources))
        if result.support_files.keys() & patched.support_files.keys():
            raise ValueError("duplicate benchmark patch support file")
        result.sources = patched.sources
        result.support_files.update(patched.support_files)
        result.notes.extend(patched.notes)
        result.patches.append(patcher.identity())
    return result
