"""Read-only compiled-instruction evidence, bound to the hardware build gate."""
from __future__ import annotations

from pathlib import Path

from . import commands, instruction_encoding
from .policy import PolicyError


def compiled_evidence(state: dict, toolbox) -> dict:
    spec = state.get('isa_spec') or {}
    if not spec.get('commands'):
        return {'checked': False, 'ok': True, 'problems': [],
                'unchecked': ['compiled instruction coverage: no frozen ISA contract']}
    gate = (state.get('gates') or {}).get('hw_build') or {}
    binding = gate.get('instruction_encoding') or {}
    rtl = binding.get('rtl') or {}
    elf = commands._software_elf(state)
    path = str(Path(elf).with_suffix('.asm')) if elf else ''
    errors = []
    if binding.get('isa_hash') and binding['isa_hash'] != instruction_encoding.digest(spec):
        errors.append('instruction_encoding/stale_contract: compiled check does not match hardware ISA')
    hw_hash = ((state.get('gates') or {}).get('implementation_hw') or {}).get('candidate_hash')
    if binding.get('candidate_hash') and hw_hash and binding['candidate_hash'] != hw_hash:
        errors.append('instruction_encoding/stale_hardware: compiled check does not match hardware candidate')
    if rtl.get('rtl_hash'):
        target = commands._soc_rtl_path(state)
        root = Path(state.get('workdir') or '.').resolve()
        try:
            rel = target.resolve().relative_to(root).as_posix() if target is not None else ''
            actual = instruction_encoding.validate_rtl(toolbox.read_text(rel, limit=10_000_000), spec)
            if actual.get('rtl_hash') != rtl['rtl_hash']:
                errors.append('instruction_encoding/stale_rtl: actual RTL changed after hardware proof')
            rtl = actual
        except (OSError, ValueError, PolicyError):
            errors.append('instruction_encoding/rtl_missing: bound RTL product is unavailable')
    if not rtl:
        # Legacy/standalone fixtures may have no gate; inspect real RTL if present.
        target = commands._soc_rtl_path(state)
        if target is not None:
            root = Path(state.get('workdir') or '.').resolve()
            try:
                rel = target.resolve().relative_to(root).as_posix()
                rtl = instruction_encoding.validate_rtl(toolbox.read_text(rel, limit=10_000_000), spec)
            except (OSError, ValueError, PolicyError):
                rtl = {}
    try:
        text = toolbox.read_text(path, limit=10_000_000) if path else ''
    except (OSError, PolicyError):
        text = ''
    if not isinstance(text, str) or not text.strip():
        # A real new hardware gate requires a compiled proof. Old standalone
        # callers without that gate remain explicitly unchecked, not certified.
        if binding:
            errors.append('instruction_encoding/assembly_missing: hardware-backed compiled disassembly is missing')
        return {'checked': False, 'ok': not errors, 'problems': errors,
                'unchecked': ['compiled instruction coverage: disassembly unavailable'],
                'assembly_path': path, 'hardware_binding': binding}
    evidence = instruction_encoding.validate_compiled_assembly(text, spec, rtl_evidence=rtl)
    if binding and not evidence.get('observed_pairs'):
        errors.append('instruction_encoding/no_custom_instruction: compiled software has no custom0 machine words')
    evidence['problems'].extend(errors)
    evidence.update(ok=not evidence['problems'], checked=evidence.get('checked', False) and not errors,
                    assembly_path=path, hardware_binding=binding)
    return evidence
