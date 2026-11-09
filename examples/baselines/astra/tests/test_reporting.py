import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from astra.reporting import FIELDS,export,validate_records
from astra.runtime import verify_package,load_config,NATIVE_REL


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=PACKAGE/'results/astra200_30min/episode_records.jsonl'
        cls.records=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        cls.definition=json.loads((PACKAGE/'configs/metric_definition.json').read_text())

    def test_primary_success_keeps_all_timeouts_and_does_not_use_endpoint(self):
        with tempfile.TemporaryDirectory() as td:
            m=export(self.records,td,self.definition)
            self.assertEqual(m['SR']['successes'],112)
            self.assertEqual(m['SR']['denominator'],200)
            self.assertEqual(m['SR']['value'],.56)
            self.assertEqual(m['timeout_cases'],88)
            self.assertEqual(m['endpoint']['successes'],106)
            self.assertEqual(m['endpoint']['observed'],112)
            self.assertIsNone(m['endpoint']['total_cohort_endpoint_rate'])

    def test_shared_csv_and_json_reconcile(self):
        with tempfile.TemporaryDirectory() as td:
            export(self.records,td,self.definition)
            with (Path(td)/'master_records.csv').open() as f:
                reader=csv.DictReader(f);rows=list(reader)
                self.assertEqual(reader.fieldnames,FIELDS)
            self.assertEqual(len(rows),40)
            self.assertEqual(sum(int(r['episodes']) for r in rows),200)
            self.assertEqual({r['num_tasks'] for r in rows},{'0'})
            summary=json.loads((Path(td)/'summary.json').read_text())
            self.assertEqual(summary['seen']['0']['astra']['overall']['SR'],.44)
            self.assertEqual(summary['unseen']['0']['astra']['overall']['SR'],.68)
            self.assertEqual(summary['seen']['0']['astra']['overall']['SubSR'],.6464)
            self.assertEqual(summary['unseen']['0']['astra']['overall']['SubSR'],.7082)

    def test_duplicates_are_not_counted_as_new_cases(self):
        rows=copy.deepcopy(self.records);rows[-1]=rows[0]
        with self.assertRaises(ValueError):validate_records(rows)

    def test_partial_or_pooled_continuation_is_rejected(self):
        with self.assertRaises(ValueError):validate_records(self.records[:-1])
        with self.assertRaises(ValueError):validate_records(self.records+self.records[:52])

    def test_unconfirmed_subsr_is_null_not_zero_or_peak(self):
        with tempfile.TemporaryDirectory() as td:
            export(self.records,td,None)
            s=json.loads((Path(td)/'summary.json').read_text())
            self.assertIsNone(s['seen']['0']['astra']['overall']['SubSR'])
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):export(self.records,td,{'field':'peak_normalized_reward'})

    def test_model_and_seed_mismatch_are_rejected(self):
        rows=copy.deepcopy(self.records);rows[0]['model']='another-model'
        with self.assertRaises(ValueError):validate_records(rows)
        rows=copy.deepcopy(self.records);rows[0]['seed']=99
        with self.assertRaises(ValueError):validate_records(rows)

    def test_native_policy_snapshot_is_unchanged(self):
        self.assertGreater(verify_package(),10)

    def test_runtime_cannot_be_written_elsewhere_inside_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive=root/'archive';native=archive/NATIVE_REL
            native.mkdir(parents=True);(native/'run_episode.py').write_text('# fixture\n')
            credential=root/'credential.json';credential.write_text('{}')
            human=root/'human';human.mkdir()
            cfg=dict(archive_root=str(archive),work_root=str(PACKAGE.parent/'forbidden_runtime'),
                python_executable=sys.executable,credentials_file=str(credential),human_root=str(human))
            config=root/'config.json';config.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load_config(config)


if __name__=='__main__':unittest.main()
