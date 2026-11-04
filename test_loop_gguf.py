import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from loop_gguf import GGUF, Tensor, align, convert, make_field, make_plan
import manage_looped


LIBRARY = Path.home() / '.lmstudio/extensions/backends/llama.cpp-linux-x86_64-vulkan-avx2-2.67.0/libggml-base.so'


def fixture(path):
    fields = [make_field('general.architecture', 8, 'qwen2'),
              make_field('qwen2.block_count', 4, 4),
              make_field('tokenizer.chat_template', 8, '{{ messages }}')]
    tensors = [Tensor(f'blk.{i}.test.weight', (1024,), 0, i * 4096) for i in range(4)]
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(fields))
    header += b''.join(f.raw for f in fields) + b''.join(t.descriptor() for t in tensors)
    header += bytes(align(len(header), 32) - len(header))
    # Different bytes in each layer catch ordering and offset mistakes.
    path.write_bytes(header + b''.join(struct.pack('<f', i + 1) * 1024 for i in range(4)))


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.converter-test-', dir=Path.cwd())
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.gguf'
        self.output = self.root / 'looped.gguf'
        fixture(self.source)
        self.before = self.source.read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def run_conversion(self, **kwargs):
        return convert(self.source, self.output, 1, 3, 2, 'Test Looped Experimental', LIBRARY, min_free=0, **kwargs)

    def test_stock_loader_accepts_and_all_layer_bytes_match(self):
        result = self.run_conversion()
        g = GGUF(self.output)
        self.assertEqual(g.layers, 6)
        self.assertEqual(g.values['tokenizer.chat_template'], '{{ messages }}')
        expected = [1, 2, 3, 2, 3, 4]
        with self.output.open('rb') as f:
            for tensor in g.tensors:
                layer = int(tensor.name.split('.')[1])
                f.seek(g.data_offset + tensor.offset)
                self.assertEqual(f.read(4096), struct.pack('<f', expected[layer]) * 1024)
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertEqual(result['validation']['installed_gguf_loader'], 'passed')
        self.assertGreater(result['storage']['shared_extent_bytes'], 0)
        self.output.unlink()
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b'user content')
        with self.assertRaises(FileExistsError):
            self.run_conversion()
        self.assertEqual(self.output.read_bytes(), b'user content')

    def test_failed_reflink_does_not_fall_back_to_copy(self):
        with patch('loop_gguf.fcntl.ioctl', side_effect=OSError('reflink unavailable')):
            with self.assertRaises(OSError):
                self.run_conversion()
        self.assertEqual(list(self.root.iterdir()), [self.source])
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_storage_cap_rejects_before_writing_output(self):
        with self.assertRaisesRegex(ValueError, 'storage cap'):
            self.run_conversion(max_literal=1)
        self.assertFalse(self.output.exists())

    def test_rejects_hybrid_architecture_and_boundary_window(self):
        model = GGUF(self.source)
        with self.assertRaises(ValueError):
            make_plan(model, 0, 2, 2, 'bad window')
        model.arch = 'qwen35moe'
        with self.assertRaisesRegex(ValueError, 'not supported'):
            make_plan(model, 1, 3, 2, 'unsupported')

    def test_managed_rollback_preserves_original_and_preview_changes_nothing(self):
        managed = self.root / 'models'
        identifier = 'test-looped'
        self.output = managed / identifier / 'test-looped.gguf'
        result = self.run_conversion()
        link = self.root / 'app-link.gguf'
        link.symlink_to(self.output)
        result.update({'identifier': identifier, 'lmstudio_link': str(link),
                       'output_mtime_ns': self.output.stat().st_mtime_ns})
        manifest = self.output.with_suffix('.gguf.json')
        manifest.write_text(json.dumps(result))
        with patch.object(manage_looped, 'MODELS', managed), patch.object(manage_looped, 'ROOT', self.root), \
             patch('manage_looped.subprocess.run', return_value=SimpleNamespace(stdout='[]', returncode=0)):
            plan = manage_looped.rollback(identifier)
            self.assertFalse(plan['apply'])
            self.assertTrue(self.output.is_file())
            self.assertTrue(link.is_symlink())
            manage_looped.rollback(identifier, apply=True)
        self.assertFalse(self.output.exists())
        self.assertFalse(link.is_symlink())
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertEqual(json.loads(manifest.read_text())['status'], 'rolled_back')

    def test_rollback_refuses_replaced_application_link(self):
        managed = self.root / 'models'
        self.output = managed / 'test-looped' / 'test-looped.gguf'
        result = self.run_conversion()
        link = self.root / 'app-link.gguf'
        link.symlink_to(self.source)
        result.update({'identifier': 'test-looped', 'lmstudio_link': str(link),
                       'output_mtime_ns': self.output.stat().st_mtime_ns})
        self.output.with_suffix('.gguf.json').write_text(json.dumps(result))
        with patch.object(manage_looped, 'MODELS', managed), patch.object(manage_looped, 'ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'link has changed'):
                manage_looped.rollback('test-looped', apply=True)
        self.assertTrue(self.output.exists())
        self.assertTrue(self.source.exists())


if __name__ == '__main__':
    unittest.main()
