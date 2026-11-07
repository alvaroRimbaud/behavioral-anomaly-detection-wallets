"""Pruebas unitarias de invariantes metodológicos del protocolo T12."""
import unittest
import numpy as np
from src.evaluation import paired_cluster_bootstrap_pr_auc, precision_at_alert_budgets, choose_threshold
from src.models import iforest_scores, set_seed

class TestMethodology(unittest.TestCase):
    def test_cluster_bootstrap_unit(self):
        y=np.array([0,1,0,1,0,1,0,1,0,1,0,1])
        u=np.repeat(np.arange(6),2)
        a=np.array([0.1,0.9,0.3,0.8,0.2,0.7,0.4,0.9,0.5,0.6,0.1,0.8])
        b=np.array([0.6,0.5,0.7,0.2,0.8,0.4,0.5,0.3,0.6,0.2,0.9,0.2])
        r=paired_cluster_bootstrap_pr_auc(y,a,b,u,n_bootstrap=100,seed=42)
        self.assertEqual(r["bootstrap_unit"],"user")
        self.assertEqual(r["n_clusters"],6)
        self.assertGreater(r["n_bootstrap_valid"],0)
        self.assertGreaterEqual(r["ci95_high"],r["ci95_low"])
    def test_alert_budget(self):
        y=np.array([0,1,0,1,0,0,0,0,0,0])
        sc=np.array([.1,.95,.3,.9,.8,.2,.6,.4,.5,.7])
        p=precision_at_alert_budgets(y,sc,budgets=[.2])
        self.assertEqual(int(p.iloc[0]["top_k"]),2)
        self.assertEqual(float(p.iloc[0]["precision_at_k"]),1.0)
    def test_threshold_only_from_validation(self):
        y=np.array([0,1,0,1,0,0,0,0,0,0])
        sc=np.array([0.1,0.8,0.2,0.9,0.3,0.4,0.2,0.1,0.5,0.4])
        t,f=choose_threshold(y,sc)
        self.assertTrue(np.isfinite(t))
        self.assertGreaterEqual(f,0)
if __name__=="__main__": unittest.main()
