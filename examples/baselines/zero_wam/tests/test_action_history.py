"""Exercise the released noise/loss methods on CPU, without loading a model."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from einops import rearrange
import torch
import torch.nn.functional as F


def native_methods():
    path = Path(__file__).resolve().parents[1] / "zero_wam/wan_va/train.py"
    tree = ast.parse(path.read_text())
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    methods = [n for n in trainer.body if isinstance(n, ast.FunctionDef)
               and n.name in ("_add_noise", "compute_loss")]
    for method in methods:
        method.decorator_list = []
    namespace = dict(
        torch=torch, F=F, rearrange=rearrange,
        _safe_mean=lambda x: x.sum() / max(x.numel(), 1),
        sample_timestep_id=lambda batch_size, **kwargs: torch.zeros(batch_size, dtype=torch.long),
        get_mesh_id=lambda f, h, w, **kwargs: torch.zeros(4, f*h*w, dtype=torch.long),
        data_seq_to_patch=lambda patch, x, f, h, w, batch_size: rearrange(
            x, "b (f h w) c -> b c f h w", f=f, h=h, w=w),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace





class UpstreamEpisodeAlgorithmTests(unittest.TestCase):
    def test_native_algorithm_definitions_match_released_commit(self):
        # Fingerprints of pristine git show08e2c4a:wan_va/train.py (AST, no formatting).
        import hashlib
        path = Path(__file__).resolve().parents[1]/"zero_wam/wan_va/train.py"
        trainer = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef))
        import json
        def stable(value):
            # Python3.13 changed ast.dump's empty-field display; omit empty
            # fields explicitly for a cross-version algorithm fingerprint.
            if isinstance(value, ast.AST):
                return [type(value).__name__, {k:stable(v) for k,v in ast.iter_fields(value) if v is not None and v != []}]
            if isinstance(value, list):
                return [stable(v) for v in value]
            return value
        expected = {'_add_noise': 'ed1fb6298e947008c9465f9a2a90b5664de2b0292255316c288534001e58a2be', '_prepare_input_dict': 'c8e8acd09a03a92ada3de517a4e94bd2ad1c85b0e88820980e0d8499e7bdc169', 'compute_loss': 'eec324a899d722ff1f7fb38703f12648ebd4b839c29445b09455845518c5ec22', '_train_step': '7522d4c074f60fe297158f0f4e44683c38d4d8124197346346fed3916fbda50f'}
        for node in trainer.body:
            if isinstance(node, ast.FunctionDef) and node.name in expected:
                self.assertEqual(hashlib.sha256(json.dumps(stable(node),sort_keys=True).encode()).hexdigest(),
                    expected.pop(node.name))
        self.assertFalse(expected)

    def test_official_mcp_shifts_mask_only_targets_outside_episode(self):
        from examples.baselines.zero_wam.zero_wam.wan_va.mcp import shift_latents_for_mcp
        source = torch.arange(13).reshape(1,1,13,1,1).float()
        for chunk in [1,2,3,4]:
            for depth in range(4):
                shift = (1+depth*2)*chunk
                target, mask = shift_latents_for_mcp(source, shift)
                count = max(0,13-shift)
                self.assertEqual(int(mask.sum()),count)
                self.assertTrue(torch.equal(target[:,:,:count],source[:,:,shift:]))
                self.assertFalse(bool(mask[:,:,count:].any()))


if __name__ == "__main__":
    unittest.main()
