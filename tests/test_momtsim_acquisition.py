"""Local metadata/hash tests; these are not a substitute for original CSV."""
import tempfile
import unittest
from pathlib import Path
from download_momtsim_original import parse_catalog, provenance

class TestPublisherSource(unittest.TestCase):
    def test_catalog_only_csv(self):
        files=parse_catalog({'results':[{'filename':'a.csv','id':'123','content_details':{'sha256_hash':'abc'}}, {'filename':'notes.txt'}]})
        self.assertEqual(len(files),1)
        self.assertEqual(files[0]['source_sha256'],'abc')
    def test_local_provenance_requires_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.csv'
            p.write_text('x,y\n1,2\n')
            data=provenance(p)
            self.assertEqual(len(data['sha256']),64)
            self.assertFalse(data['originality_attested_by_researcher'])
            with self.assertRaises(ValueError):
                provenance(p,{'source_sha256':'0'*64})
if __name__=='__main__':unittest.main()
