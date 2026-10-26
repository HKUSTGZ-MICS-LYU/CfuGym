"""Fast whole-SoC Vivado contracts. Never invokes Vivado or a real model."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent.runtime import vivado

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'skills/cfu-designer/scripts/vivado_ppa_report.py'
TCL = SCRIPT.with_suffix('.tcl').with_name('vivado_ppa.tcl')
SPEC = importlib.util.spec_from_file_location('_test_vivado_ppa_driver', SCRIPT)
assert SPEC and SPEC.loader
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)

AREA = '''
| Site Type | Used | Fixed | Available |
| Slice LUTs* | 123 | 0 | 133800 |
| LUT as Distributed RAM | 5 | 0 | |
| Slice Registers | 234 | 0 | 267600 |
| Slice Registers | 234 | 0 | 267600 |
| Block RAM Tile | 1.5 | 0 | 365 |
| RAMB36/FIFO* | 1 | 0 | 365 |
| RAMB36E1 only | 1 | | |
| RAMB18 | 1 | 0 | 730 |
| RAMB18E1 only | 1 | | |
| DSPs | 2 | 0 | 740 |
| DSP48E1 only | 2 | | |
'''
POWER = '''
| Total On-Chip Power (W) | 0.250 |
| Dynamic (W) | 0.100 |
| Device Static (W) | 0.150 |
| Confidence Level | Medium |
3.1 By Hierarchy
----------------
| Name                | Power (W) |
| MiCoSoc             | 0.100 |
|   cpu               | 0.070 |
|   container         | 0.030 |
|     agentCfu        | <0.001 |
'''

# Shape copied from Vivado 2025.2 CRC32 post-route power.rpt: the SoC total
# exists, but the leaf AgentCfu hierarchy is omitted from the table.
POWER_NO_CFU_HIERARCHY = '''
| Dynamic (W) | 0.064 |
| Device Static (W) | 0.474 |
| Confidence Level | Medium |
3.1 By Hierarchy
----------------
| Name | Power (W) |
| MiCoSoc | 0.064 |
|   socCtrl_systemClk_IBUF_inst | 0.002 |
|   system_cpu_logic_core | 0.014 |
|   system_mainBus_arbiter_core | 0.003 |
|   system_peripheral_uart_logic_core | 0.001 |
|   system_ram_thread_logic | 0.040 |
'''


def metrics_fixture() -> dict[str, str]:
    result = {'vivado.version': '2025.2', 'part': vivado.DEFAULT_PART,
              'target_frequency_mhz': '100.0', 'phase': 'post_route',
              'cfu.instance_count': '1', 'cfu.instance': 'container/agentCfu',
              'soc.unconstrained_internal_endpoints': '0', 'soc.no_clock': '0',
              'route.fully_routed': '1', 'route.errors_in_routes': '0', 'route.fully_placed': '1',
              'drc.errors': '0', 'drc.critical_warnings': '0', 'drc.warnings': '1'}
    for prefix in ['soc', *['cfu.' + name for name in vivado.PATH_GROUPS]]:
        for mode in ('setup', 'hold'):
            key = prefix + '.' + mode
            result.update({key + '.exists': '1', key + '.slack_ns': '0.150',
                           key + '.datapath_delay_ns': '3.500', key + '.startpoint': 'reg_a/Q',
                           key + '.endpoint': 'reg_b/D', key + '.total_negative_slack_ns': '0.0',
                           key + '.failing_endpoints': '0'})
    return result


def make_summary(metrics: dict[str, str] | None = None, **kwargs):
    power_text = kwargs.pop('power', POWER)
    return vivado.make_summary(metrics or metrics_fixture(), AREA, AREA, power_text,
                                returncode=kwargs.pop('returncode', 0),
                                soc_synthesis_area=AREA, cfu_synthesis_area=AREA, **kwargs)


class VivadoParsingTest(unittest.TestCase):
    def test_finite_numbers_null_and_bound_power(self):
        for value in ('NA', None, '', 'NaN', 'inf', '-inf', '<0.001'):
            self.assertIsNone(vivado.number(value))
        self.assertEqual(vivado.number('1,234.5'), 1234.5)
        bound = vivado.power_value('<0.001')
        self.assertIsNone(bound['value'])
        self.assertEqual(bound['upper_bound'], 0.001)
        self.assertTrue(bound['bound_exclusive'])
        self.assertIsNone(vivado.power_value('-0.1')['value'])
        self.assertIsNone(vivado.power_value('<NA')['upper_bound'])

    def test_fractional_bram_repeated_rows_and_dsp_aliases(self):
        area = vivado.parse_utilization(AREA)
        self.assertEqual(area, {'lut': 123.0, 'ff': 234.0, 'bram_36k': 1.5,
                                'dsp': 2.0, 'lutram': 5.0, 'ramb18': 1.0, 'ramb36': 1.0})
        self.assertIsNone(vivado.parse_utilization('')['bram_36k'])
        self.assertIsNone(vivado.parse_utilization(AREA + '| Slice Registers | 99 |\n')['ff'])

    def test_synthesized_zero_lutram_subrow_omitted_in_real_2025_2_report(self):
        # Reduced table from the real whole-MiCoSoc acceptance report. Vivado
        # omits the distributed-RAM/SRL subrows when LUT as Memory is zero.
        report = '''
| Slice LUTs*             | 250 | 0 | 134600 |
| LUT as Logic           | 250 | 0 | 134600 |
| LUT as Memory          |   0 | 0 | 46200 |
| Slice Registers        | 542 | 0 | 269200 |
| Block RAM Tile         |   0 | 0 | 365 |
| RAMB36/FIFO*           |   0 | 0 | 365 |
| RAMB18                 |   0 | 0 | 730 |
| DSPs                   |   0 | 0 | 740 |
'''
        self.assertEqual(vivado.parse_utilization(report)['lutram'], 0.0)
        self.assertIsNone(vivado.parse_utilization(report.replace('|   0 | 0 | 46200 |',
                                                                   '|   1 | 0 | 46200 |'))['lutram'])
        partial = AREA.replace('| LUT as Distributed RAM | 5 | 0 | |\n', '')
        result = vivado.make_summary(metrics_fixture(), partial, partial, POWER, returncode=0,
                                     soc_synthesis_area=partial, cfu_synthesis_area=partial)
        self.assertEqual(result['status'], 'passed', result['issues'])
        self.assertIsNone(result['cfu']['synthesis_area']['lutram'])
        self.assertTrue(any(i['code'] == 'missing_auxiliary_area' for i in result['issues']))
        self.assertIsNone(vivado.parse_utilization('')['lutram'])

    def test_exact_dynamic_hierarchy_without_static_apportionment(self):
        power = vivado.parse_power(POWER, 'container/agentCfu')
        self.assertEqual(power['soc_dynamic']['value'], 0.100)
        self.assertEqual(power['soc_static']['value'], 0.150)
        self.assertEqual(power['cfu_dynamic']['upper_bound'], 0.001)
        self.assertIsNone(vivado.parse_power(POWER, 'agentCfu')['cfu_dynamic']['upper_bound'])
        duplicate = POWER + '|     agentCfu | 0.005 |\n'
        self.assertIsNone(vivado.parse_power(duplicate, 'container/agentCfu')['cfu_dynamic']['value'])

    def test_omitted_cfu_leaf_uses_only_conservative_soc_dynamic_upper_bound(self):
        result = make_summary(power=POWER_NO_CFU_HIERARCHY)
        cfu_power = result['cfu']['power']
        self.assertEqual(result['status'], 'passed', result['issues'])
        self.assertIsNone(cfu_power['dynamic']['value'])
        self.assertEqual(cfu_power['dynamic']['upper_bound'], 0.064)
        self.assertEqual(cfu_power['dynamic']['reported'], '<= SoC dynamic total (0.064 W)')
        self.assertEqual(cfu_power['method'], 'whole_soc_dynamic_power_upper_bound')
        markdown = vivado.render_markdown(result)
        self.assertIn('≤0.064', markdown)
        self.assertIn('not a CFU point estimate', markdown)

    def test_missing_soc_dynamic_does_not_fabricate_cfu_bound(self):
        result = make_summary(power='3.1 By Hierarchy\n| Name | Power (W) |\n| MiCoSoc | 0.01 |')
        self.assertIsNone(result['cfu']['power']['dynamic']['upper_bound'])
        self.assertIn('missing_dynamic_power', [issue['code'] for issue in result['issues']])

    def test_machine_tsv_strict_and_phase_progress(self):
        self.assertEqual(vivado.parse_metrics_tsv('phase\tpost_synth\nphase\tpost_route\n'),
                         {'phase': 'post_route'})
        for bad in ('a\t1\na\t2', 'a\tb\tc', 'key with spaces\tx'):
            with self.assertRaises(ValueError):
                vivado.parse_metrics_tsv(bad)

    def test_pass_contains_real_target_no_fmax_and_static_is_soc_only(self):
        result = make_summary()
        self.assertEqual(result['status'], 'passed', result['issues'])
        self.assertEqual(result['backend'], 'vivado')
        self.assertIsNone(result['cfu']['power']['device_static'])
        self.assertEqual(result['soc']['synthesis_area']['bram_36k'], 1.5)
        self.assertNotIn('fmax', json.dumps(result).lower())
        self.assertIn('No Fmax is inferred', vivado.render_markdown(result))
        self.assertEqual(result['soc']['timing']['setup']['total_negative_slack_method'],
                         'sum_of_reported_worst_endpoint_slacks')
        self.assertIn('already rounded', vivado.render_markdown(result))

    def test_zero_exit_does_not_override_failed_closure_or_missing_metrics(self):
        mutations = {'soc.setup.slack_ns': '-0.001', 'soc.hold.slack_ns': '-0.001',
                     'soc.setup.total_negative_slack_ns': '-1.0', 'soc.hold.failing_endpoints': '1',
                     'soc.no_clock': '1', 'soc.unconstrained_internal_endpoints': '1',
                     'route.fully_routed': '0', 'route.errors_in_routes': '1',
                     'drc.errors': '1', 'drc.critical_warnings': '1', 'cfu.instance_count': '2',
                     'phase': 'post_synth', 'cfu.internal.setup.slack_ns': 'NA'}
        for key, value in mutations.items():
            with self.subTest(key=key):
                metrics = metrics_fixture()
                metrics[key] = value
                self.assertEqual(make_summary(metrics)['status'], 'failed')
        for key in metrics_fixture():
            with self.subTest(missing=key):
                metrics = metrics_fixture()
                del metrics[key]
                self.assertEqual(make_summary(metrics)['status'], 'failed')
        self.assertEqual(make_summary(returncode=1)['status'], 'failed')

    def test_only_exact_ucio1_critical_is_part_only_board_readiness_warning(self):
        metrics = metrics_fixture()
        metrics.update({'drc.critical_warnings': '1', 'drc.board_unbound.rule_code': 'UCIO-1',
                        'drc.board_unbound.raw_severity': 'Critical Warning',
                        'drc.board_unbound.scope': 'part_only_no_bitstream',
                        'drc.board_unbound.critical_warnings': '1'})
        result = make_summary(metrics)
        self.assertEqual(result['status'], 'passed', result['issues'])
        self.assertEqual(result['soc']['drc']['critical_warnings'], 1)
        self.assertEqual(result['soc']['drc']['implementation_critical_warnings'], 0)
        board = result['soc']['board_readiness']
        self.assertEqual(board['status'], 'unbound')
        self.assertFalse(board['bitstream_ready'])
        self.assertEqual(board['exceptions'][0]['raw_severity'], 'Critical Warning')
        self.assertTrue(any(i['code'] == 'board_unbound' and i['severity'] == 'warning'
                            for i in result['issues']))
        mutations = {'drc.board_unbound.rule_code': 'UCIO-10',
                     'drc.board_unbound.raw_severity': 'Error',
                     'drc.board_unbound.scope': 'board_validated',
                     'drc.board_unbound.critical_warnings': '0.5',
                     'drc.errors': '1', 'drc.critical_warnings': '2',
                     'soc.setup.slack_ns': '-0.001', 'soc.hold.slack_ns': '-0.001'}
        for key, value in mutations.items():
            with self.subTest(key=key):
                changed = {**metrics, key: value}
                self.assertEqual(make_summary(changed)['status'], 'failed')
        # Historical metrics must not retroactively acquire an exception.
        legacy = metrics_fixture()
        legacy['drc.critical_warnings'] = '1'
        result = make_summary(legacy)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['soc']['board_readiness']['classification_evidence'])
        self.assertEqual(result['soc']['drc']['implementation_critical_warnings'], 1)

    def test_combinational_cfu_has_null_internal_and_legitimate_no_paths(self):
        metrics = metrics_fixture()
        for mode in ('setup', 'hold'):
            for suffix in ('slack_ns', 'datapath_delay_ns', 'startpoint', 'endpoint',
                           'total_negative_slack_ns', 'failing_endpoints'):
                metrics.pop(f'cfu.internal.{mode}.{suffix}')
            metrics[f'cfu.internal.{mode}.exists'] = '0'
        result = make_summary(metrics)
        self.assertEqual(result['status'], 'passed')
        self.assertIsNone(result['cfu']['timing']['internal']['setup']['slack_ns'])


class VivadoInputTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vivado-contract-', dir=REPO)
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.root = self.run / 'snapshot' / 'rtl'
        self.root.mkdir(parents=True)
        self.rtl = self.root / 'MiCoSoc.v'
        self.rtl.write_text('module MiCoSoc(input socCtrl_systemClk); endmodule\n')
        self.out = self.run / '11_cost' / 'eval' / 'vivado'

    def validate(self, files=None, out=None, **kwargs):
        return vivado.validate_inputs(self.run, self.root, files or [self.rtl], out or self.out, **kwargs)

    def test_build_command_is_fixed_relative_argv_and_no_effects(self):
        memory = self.root / 'ram.bin'
        memory.write_text('0011\n')
        state = {'workdir': str(REPO), 'run_root': str(self.run.relative_to(REPO)),
                 'rtl_snapshot': {'files': [{'path': p.name, 'sha256': vivado.sha256_file(p)}
                                            for p in (self.rtl, memory)]},
                 'eda_timeout': 12, 'vivado_threads': 2}
        argv = vivado.build_command(state, self.root.relative_to(REPO), self.out.relative_to(REPO))
        self.assertEqual(argv[:2], ['python3', 'skills/cfu-designer/scripts/vivado_ppa_report.py'])
        self.assertFalse(any(arg.startswith('/') for arg in argv))
        self.assertEqual(argv.count('--rtl'), 2)
        self.assertIn('100.0', argv)
        self.assertFalse(self.out.exists())
        with self.assertRaises(ValueError):
            vivado.build_command({**state, 'rtl_snapshot': {}}, self.root, self.out)

    def test_path_confinement_and_symlink_escape(self):
        external = self.run / 'outside.v'
        external.write_text('module Other(); endmodule')
        with self.assertRaises(ValueError):
            self.validate(files=[self.rtl, external])
        with self.assertRaises(ValueError):
            self.validate(out=self.root / 'results')
        with self.assertRaises(ValueError):
            self.validate(out=self.run.parent / 'results')
        link = self.root / 'escape.v'
        link.symlink_to(external)
        with self.assertRaises(ValueError):
            self.validate(files=[self.rtl, link])

    def test_reject_simulator_standalone_and_tcl_injection(self):
        for path in (self.root / 'MiCoSocSim.v', self.root / 'simWorkspace' / 'MiCoSoc.v'):
            path.parent.mkdir(exist_ok=True)
            path.write_text(self.rtl.read_text())
            with self.assertRaises(ValueError):
                self.validate(files=[path])
        self.rtl.write_text('module AgentCfu(); endmodule')
        with self.assertRaises(ValueError):
            self.validate()
        self.rtl.write_text('module MiCoSoc(); endmodule')
        for part in ('xc7a200tsbg484-1;exit', '[exec touch /tmp/evil]', 'xc7a200t\nexit'):
            with self.assertRaises(ValueError):
                self.validate(part=part)
        for freq in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.validate(target_frequency_mhz=freq)

    def test_headers_and_memory_must_be_in_complete_fileset(self):
        header = self.root / 'defs.vh'
        memory = self.root / 'ram.bin'
        header.write_text('`define SIZE 4\n')
        memory.write_text('0011\n')
        self.rtl.write_text('`include "defs.vh"\nmodule MiCoSoc(); initial $readmemb("ram.bin",m); endmodule')
        with self.assertRaises(ValueError):
            self.validate()
        self.validate(files=[self.rtl, header, memory])
        self.rtl.write_text('module MiCoSoc(); initial $readmemb("../../outside.bin",m); endmodule')
        with self.assertRaises(ValueError):
            self.validate()
        self.rtl.write_text('module MiCoSoc(); initial $readmemb(variable,m); endmodule')
        with self.assertRaises(ValueError):
            self.validate()

    def _args(self):
        return DRIVER.parser().parse_args(['--run-dir', str(self.run), '--rtl-root', str(self.root),
                                           '--rtl', str(self.rtl), '--out-dir', str(self.out)])

    def _fake_process(self, argv, cwd, log_path, timeout):
        self.assertEqual(argv[0], 'vivado')
        self.assertEqual(Path(argv[argv.index('-source') + 1]), TCL)
        self.assertEqual(cwd, self.out)
        staged = self.out / 'inputs' / self.rtl.name
        self.assertEqual(staged.read_bytes(), self.rtl.read_bytes())
        self.assertEqual(self.rtl.read_text(), 'module MiCoSoc(input socCtrl_systemClk); endmodule\n')
        data = metrics_fixture()
        (self.out / 'metrics.tsv').write_text(''.join(f'{key}\t{value}\n' for key, value in data.items()))
        for name in ('utilization_soc.rpt', 'utilization_cfu.rpt', 'utilization_post_synth.rpt',
                     'utilization_cfu_post_synth.rpt'):
            (self.out / name).write_text(AREA)
        (self.out / 'power.rpt').write_text(POWER)
        return {'returncode': 0, 'execution_issue': None, 'elapsed_seconds': 0.001, 'cancel_signal': None}

    def test_driver_stages_sources_and_loader_reconstructs_hashed_evidence(self):
        source_hash = vivado.sha256_file(self.rtl)
        with mock.patch.object(DRIVER, 'run_process', side_effect=self._fake_process):
            summary = DRIVER.execute(self._args())
        self.assertEqual(summary['status'], 'passed', summary['issues'])
        self.assertEqual(vivado.load_summary(REPO, self.out), summary)
        self.assertEqual(vivado.sha256_file(self.rtl), source_hash)
        self.assertTrue((self.out / 'summary.md').is_file())
        with self.assertRaises(FileExistsError):
            DRIVER.execute(self._args())
        (self.out / 'metrics.tsv').write_text('phase\tpost_synth\n')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            vivado.load_summary(REPO, self.out)

    def test_failed_launch_has_parseable_partial_summary(self):
        with mock.patch.object(DRIVER, 'run_process', side_effect=FileNotFoundError('no vivado')):
            summary = DRIVER.execute(self._args())
        self.assertEqual(summary['status'], 'failed')
        self.assertIsNone(summary['soc']['area']['lut'])
        self.assertEqual(vivado.load_summary(REPO, self.out), summary)

    def test_wrong_frozen_hash_does_not_create_output(self):
        args = self._args()
        args.sha256 = ['0' * 64]
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            DRIVER.execute(args)
        self.assertFalse(self.out.exists())

    def test_summary_status_cannot_be_edited_to_pass(self):
        with mock.patch.object(DRIVER, 'run_process', side_effect=FileNotFoundError('no vivado')):
            DRIVER.execute(self._args())
        file = self.out / 'summary.json'
        summary = json.loads(file.read_text())
        summary['status'] = 'passed'
        file.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            vivado.load_summary(REPO, self.out)


@unittest.skipUnless(sys.platform == 'linux', 'Linux process-tree contract')
class VivadoCancellationTest(unittest.TestCase):
    def _cancel(self, group: bool):
        with tempfile.TemporaryDirectory(prefix='vivado-cancel-') as temp:
            root = Path(temp)
            ready = root / 'ready.json'
            pids = root / 'pids.json'
            result_path = root / 'result.json'
            code = f'''
import importlib.util,json,os,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('driver', {str(SCRIPT)!r})
driver=importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
worker="import json,os,subprocess,sys,time; c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); open({str(pids)!r},'w').write(json.dumps([os.getpid(),c.pid,os.getpgrp()])); time.sleep(60)"
# Ready is written by a child only after the parent's signal handlers are active.
result=driver.run_process([sys.executable,'-c',worker],Path({temp!r}),Path({str(root / 'log')!r}),20,grace_seconds=.2)
Path({str(result_path)!r}).write_text(json.dumps(result))
'''
            # The TEST coordinator owns a group, exactly like runtime.checks.
            process = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 8
                while not pids.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pids.is_file(), 'fake worker did not start')
                worker, grandchild, pgid = json.loads(pids.read_text())
                self.assertEqual(pgid, process.pid, 'driver must preserve coordinator process group')
                if group:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 0, stdout + stderr)
                outcome = json.loads(result_path.read_text())
                self.assertEqual(outcome['cancel_signal'], signal.SIGTERM)
                self.assertIn('cancelled', outcome['execution_issue'])
                for pid in (worker, grandchild):
                    info = DRIVER._process_info(pid)
                    self.assertTrue(info is None or info[2] == 'Z', f'live worker leaked: {pid}')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=8)

    def test_parent_process_group_cancellation_leaves_no_vivado_children(self):
        self._cancel(group=True)

    def test_direct_signal_is_forwarded_to_vivado_grandchildren(self):
        self._cancel(group=False)

    def test_timeout_kills_stubborn_worker_and_descendants(self):
        with tempfile.TemporaryDirectory(prefix='vivado-timeout-') as temp:
            root = Path(temp)
            pidfile = root / 'pids.json'
            worker = ('import subprocess,sys,signal,time,os,json; '
                      'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                      "c=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); "
                      f'open({str(pidfile)!r},"w").write(json.dumps([os.getpid(),c.pid])); time.sleep(60)')
            outcome = DRIVER.run_process([sys.executable, '-c', worker], root, root / 'log',
                                         timeout=0.2, grace_seconds=0.1)
            self.assertIn('timeout', outcome['execution_issue'])
            for pid in json.loads(pidfile.read_text()):
                info = DRIVER._process_info(pid)
                self.assertTrue(info is None or info[2] == 'Z')


class VivadoFixedTclTest(unittest.TestCase):
    def test_fixed_whole_soc_flow_and_pre_synth_clock(self):
        text = TCL.read_text()
        self.assertIn('synth_design -top MiCoSoc -part $part -flatten_hierarchy none', text)
        self.assertLess(text.index('read_xdc $xdc'), text.index('synth_design -top'))
        for command in ('opt_design', 'place_design', 'phys_opt_design', 'route_design'):
            self.assertIn('\n    ' + command + '\n', text)
        self.assertNotIn('set_property DONT_TOUCH', text)
        self.assertNotRegex(text, r'(?m)^\s*eval\s')
        self.assertNotIn('source $', text)
        self.assertIn('socCtrl_systemClk', text)
        self.assertIn('-no_report_unconstrained', text)
        self.assertIn('unconstrained_internal_endpoints', text)
        self.assertIn('ROUTED_FULLY', text)

    @unittest.skipUnless(shutil.which('tclsh'), 'Tcl interpreter unavailable')
    def test_fixed_tcl_is_syntactically_complete_without_running_vivado(self):
        script = 'set f [open $env(CFU_TCL) r]; set s [read $f]; close $f; puts [info complete $s]\n'
        outcome = subprocess.run(['tclsh'], input=script, text=True, capture_output=True,
                                 env={**os.environ, 'CFU_TCL': str(TCL)}, check=False)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        self.assertEqual(outcome.stdout.strip(), '1')


if __name__ == '__main__':
    unittest.main()
