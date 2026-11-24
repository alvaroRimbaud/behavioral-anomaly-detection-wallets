import unittest
import numpy as np
import pandas as pd
from run_external_validation import HYBRID, build_external_features, external_holdout


class TestStrictSteps(unittest.TestCase):
    def test_tied_transactions_and_labels(self):
        frame = pd.DataFrame(dict(step=[0,0,1,1,2,25], amount=[1,3,4,9,8,7],
            sender="a", recipient="b", label=[0,1]*3, _row_order=range(6),
            phase=["warmup"]*2+["train"]*3+["test"]))
        features, _ = build_external_features(frame, 2)
        np.testing.assert_equal(features.tx_count_last_1h.to_numpy(), [0,0,2,2,2,0])
        np.testing.assert_equal(features.tx_count_last_24h.to_numpy(), [0,0,2,2,4,3])
        np.testing.assert_allclose(features.time_since_last_tx.iloc[2:5], np.log(2))
        other, _ = build_external_features(frame.assign(_row_order=-frame._row_order, label=1-frame.label), 2)
        np.testing.assert_allclose(features[HYBRID], other[HYBRID])

    def test_disjoint_internal_holdout(self):
        train = pd.DataFrame({"step":[0,1,1,2,2,2]})
        fit, stop = external_holdout(train)
        self.assertLess(fit.step.max(), stop.step.min())
        self.assertEqual(len(fit)+len(stop), len(train))
