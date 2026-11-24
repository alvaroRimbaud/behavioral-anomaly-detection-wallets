"""Non-inferential tests for O1/O2 invariants, never external evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_external_validation import (
    assign_temporal_phases, build_external_features, load_native,
    RAW, HYBRID,
)
from run_multiseed import completed_seed, REQUIRED_SEED_FILES
from src.simulator import SimulationConfig, simulate_transactions
from src.evaluation import estimate_d50_from_curve


class TestExternalIndependentDataProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = Path(__file__).parent / 'fixtures' / 'INDEPENDENT_SCHEMA_SMOKE_NOT_EXTERNAL.csv'

    def test_external_schema_temporal_and_missing_features(self):
        frame, sha = load_native(self.fixture, 'momtsim')
        self.assertEqual(len(sha), 64)
        frame = assign_temporal_phases(frame)
        for a, b in [('warmup','train'),('train','validation'),('validation','test')]:
            self.assertLess(frame.loc[frame.phase==a,'step'].max(),
                            frame.loc[frame.phase==b,'step'].min())
        x, status = build_external_features(frame, min_warmup=3)
        self.assertTrue(all(c in x.columns for c in HYBRID))
        self.assertNotIn('device_novelty_user', HYBRID)
        self.assertEqual((len(RAW), len(HYBRID)), (6,10))
        self.assertGreater(status['retained_rows'], 0)

    def test_ground_truth_cannot_modify_profiles_or_features(self):
        frame, _ = load_native(self.fixture, 'momtsim')
        frame = assign_temporal_phases(frame)
        a,_ = build_external_features(frame, min_warmup=3)
        altered = frame.copy()
        # Deliberately flip all original labels; feature engineering must be invariant.
        altered['label'] = 1 - altered['label']
        b,_ = build_external_features(altered, min_warmup=3)
        np.testing.assert_allclose(a[HYBRID].to_numpy(float), b[HYBRID].to_numpy(float))


class TestInternalExperimentIntegrity(unittest.TestCase):
    def test_target_intensity_bands_balanced_by_phase(self):
        df, _, _ = simulate_transactions(SimulationConfig(n_users=45, seed=67))
        for phase in ('validation','test'):
            d = df[(df.phase == phase) & (df.is_anomaly == 1)]
            counts = d.target_intensity_band.value_counts().to_dict()
            self.assertEqual(set(counts), {'baja','media','alta'})
            self.assertLessEqual(max(counts.values())-min(counts.values()),1)

    def test_sparse_d50_is_not_estimable(self):
        sparse = pd.DataFrame({'D_mid':[.3,.5,.7], 'recall':[.2,.6,.9],
                               'n_anomalies':[4,5,2]})
        r = estimate_d50_from_curve(sparse)
        self.assertTrue(np.isnan(r['d50']))
        self.assertNotEqual(r['d50_status'],'estimado_exploratorio_con_soporte')

    def test_resume_checks_files_and_compatible_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            self.assertFalse(completed_seed(out,seed=42,users=500,tuning='full',bootstrap=1000))
            for name in REQUIRED_SEED_FILES:
                (out/name).write_text('x')
            (out/'run_summary.json').write_text(json.dumps({
                'seed':42,'n_users':500,'tuning_profile':'full',
                'bootstrap_iterations':1000,'bootstrap_unit':'user',
                'train_anomalies':0,'raw_features':6,'hybrid_features':12
            }))
            self.assertTrue(completed_seed(out,seed=42,users=500,tuning='full',bootstrap=1000))
            self.assertFalse(completed_seed(out,seed=42,users=1000,tuning='full',bootstrap=1000))
            (out/'scores_test_all_models.csv').unlink()
            self.assertFalse(completed_seed(out,seed=42,users=500,tuning='full',bootstrap=1000))

if __name__ == '__main__':
    unittest.main()
