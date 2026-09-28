"""Security regressions use temporary repositories, never historical runs."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.cfu_design_agent.runtime.tools import RepoToolbox, validate_text_patch
from agents.cfu_design_agent.runtime.policy import load_policy, PolicyError
from agents.cfu_design_agent.runtime.agent_tools import ReadOnlyToolset
from agents.tests.common import REPO_ROOT


class HardeningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = 'agents/generated/runs/hardening'
        self.target = self.run + '/workspace/design/AgentCfu.scala'
        self.box = RepoToolbox(self.root, policy=load_policy(REPO_ROOT), stage='implementation',
                               run_root=self.run, apply_patch=True)
        self.put(self.target, 'old\n')

    def put(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)
        return p

    def patch(self, path=None):
        path = path or self.target
        return f'--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n'

    def test_read_scope_blocks_other_run_and_prefix_sibling(self):
        for run in ('other', 'hardening-sibling'):
            path = f'agents/generated/runs/{run}/workspace/design/AgentCfu.scala'
            self.put(path, 'other-run fixture')
            self.assertFalse(self.box.is_read_allowed(path))
            with self.assertRaises(PolicyError):
                self.box.read_text(path)
            result = ReadOnlyToolset(self.box).run('read_file', {'path': path})
            self.assertTrue(result.startswith('denied by access policy:'), result)
        self.assertEqual(self.box.read_text(self.target), 'old\n')
        self.box.run_root = ''
        self.assertEqual(self.box.read_text(path), 'other-run fixture')

    def test_tools_do_not_list_or_search_other_run(self):
        other = 'agents/generated/runs/other/workspace/design/AgentCfu.scala'
        self.put(other, 'private_run_marker')
        tools = ReadOnlyToolset(self.box)
        for name, args in [('list_dir', {'path': 'agents/generated/runs/other'}),
                           ('search_text', {'path': 'agents/generated/runs/other', 'pattern': 'marker'})]:
            self.assertTrue(tools.run(name, args).startswith('denied by access policy:'))
        listing = tools.run('list_dir', {'path': 'agents/generated/runs', 'glob': '**/*'})
        self.assertNotIn('/other/', listing)
        self.assertIn(self.target, listing)
        search = tools.run('search_text', {'path': 'agents/generated/runs', 'pattern': 'private_run_marker'})
        self.assertEqual(search, '(no matches)')

    def test_tracked_read_still_redirects_to_current_run(self):
        tracked = 'src/main/scala/vexiiriscv/soc/mico/AgentCfu.scala'
        self.put(tracked, 'tracked fixture')
        result = ReadOnlyToolset(self.box).run('read_file', {'path': tracked})
        self.assertIn('old', result)
        self.assertNotIn('tracked fixture', result)

    def test_patch_rejects_metadata_and_ambiguous_headers(self):
        cases = [
            '', 'diff --git a/x b/x\nold mode 100644\nnew mode 100755\n',
            'diff --git a/x b/y\nsimilarity index 100%\nrename from x\nrename to y\n',
            'diff --git a/x b/y\ncopy from x\ncopy to y\n',
            'diff --git a/x b/x\nGIT binary patch\nliteral 0\n',
            'diff --git a/x b/x\nnew file mode 120000\n' + self.patch('x'),
            self.patch('"quoted"'),
            'diff --git a/wrong b/wrong\n' + self.patch(),
            self.patch() + self.patch(),
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_text_patch(text)

    def test_regular_diff_headers_and_index_are_supported(self):
        text = f'diff --git a/{self.target} b/{self.target}\nindex abc..def 100644\n' + self.patch()
        self.assertEqual(validate_text_patch(text), [self.target])

    def test_patch_other_run_and_tests_rejected_before_execution(self):
        for path in ['sw/tests/foo.c', 'agents/generated/runs/other/workspace/a_benchmark/foo.c',
                     self.run + '/run.yaml']:
            with mock.patch.object(self.box, 'run_command') as command:
                with self.assertRaises(ValueError):
                    self.box.apply_patch_text(self.patch(path))
                command.assert_not_called()

    def test_real_patch_applies_without_unsafe_flag(self):
        import subprocess
        real = subprocess.run
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            return real(args, **kwargs)
        with mock.patch('agents.cfu_design_agent.runtime.tools.subprocess.run', side_effect=run):
            result = self.box.apply_patch_text(self.patch())
        self.assertEqual(result['returncode'], 0, result)
        self.assertEqual((self.root / self.target).read_text(), 'new\n')
        self.assertTrue(all('--unsafe-paths' not in args for args in calls))

    def test_file_prevalidation_and_duplicate_rejection(self):
        for blocks in [[(self.target, 'new'), ('sw/tests/no.c', 'bad')],
                       [(self.target, 'one'), (self.target, 'two')]]:
            self.assertEqual(self.box.apply_file_blocks(blocks)['returncode'], 1)
            self.assertEqual((self.root / self.target).read_text(), 'old\n')

    def test_file_rollback_on_later_write_failure(self):
        second = self.run + '/workspace/design/AgentCfuFiber.scala'
        self.put(second, 'second')
        real = self.box.write_text
        def write(path, content):
            if path == second:
                raise OSError('injected failure')
            return real(path, content)
        with mock.patch.object(self.box, 'write_text', side_effect=write):
            result = self.box.apply_file_blocks([(self.target, 'new'), (second, 'changed')])
        self.assertEqual(result['returncode'], 1)
        self.assertEqual((self.root / self.target).read_text(), 'old\n')
        self.assertEqual((self.root / second).read_text(), 'second')

    def test_copy_rejects_symlink_and_denied_children_before_writing(self):
        src = 'agents/examples/project'
        dst = self.run + '/workspace/a_project_profile'
        self.put(src + '/ok.c', 'ok')
        hidden = self.put('private.txt', 'not a secret, synthetic fixture')
        (self.root / src / 'normal.c').symlink_to(hidden)
        self.box.stage = 'prepare_benchmark'
        with self.assertRaises(PolicyError):
            self.box.copy_tree(src, dst)
        self.assertFalse((self.root / dst).exists())
        (self.root / src / 'normal.c').unlink()
        self.put(src + '/.env', 'synthetic')
        with self.assertRaises(PolicyError):
            self.box.copy_tree(src, dst)
        self.assertFalse((self.root / dst).exists())

    def test_copy_root_symlink_rejected(self):
        self.put('agents/examples/real/a.c', 'fixture')
        (self.root / 'agents/examples/alias').symlink_to(self.root / 'agents/examples/real')
        self.box.stage = 'prepare_benchmark'
        with self.assertRaises(PolicyError):
            self.box.copy_tree('agents/examples/alias', self.run + '/workspace/a_project_profile')

    def test_copy_valid_tree(self):
        self.put('agents/examples/project/a.c', 'fixture')
        self.box.stage = 'prepare_benchmark'
        dst = self.run + '/workspace/a_project_profile'
        self.box.copy_tree('agents/examples/project', dst)
        self.assertEqual((self.root / dst / 'a.c').read_text(), 'fixture')

    def test_read_requests_bounded_char_count(self):
        stream = io.StringIO('abcdefghij')
        with mock.patch.object(Path, 'open', return_value=stream), mock.patch.object(stream, 'read', wraps=stream.read) as read:
            self.assertEqual(self.box.read_text(self.target, limit=3), 'abc')
            read.assert_called_once_with(3)

    def test_unsafe_regex_and_enumeration_fail_safely(self):
        tools = ReadOnlyToolset(self.box, max_files=1)
        result = tools.run('search_text', {'pattern': '(a+)+$', 'path': self.run})
        self.assertTrue(result.startswith('error:'), result)
        for i in range(25):
            self.put(self.run + f'/hidden{i}.txt', 'x')
        result = tools.run('list_dir', {'path': self.run, 'glob': '**/*'})
        self.assertIn('budget exceeded', result)

    def test_error_observation_is_not_success(self):
        from agents.tests.test_agent_tools import _FakeChatClient, _tool_call_response, _final_response
        from agents.cfu_design_agent.runtime.llm import OpenAiTextClient
        fake = _FakeChatClient([_tool_call_response('read_file', '{}'), _final_response('done')])
        client = OpenAiTextClient(config={'model': 'm', 'api_key': 'synthetic', 'api_mode': 'chat'})
        with mock.patch('openai.OpenAI', return_value=fake):
            result = client.complete_with_tools('s', 'u', tools=[], max_rounds=1,
                tool_runner=lambda *_: 'denied by access policy: synthetic')
        self.assertFalse(result.tool_calls[0]['ok'])


if __name__ == '__main__':
    unittest.main()
