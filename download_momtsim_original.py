"""Retrieve ORIGINAL MoMTSim v2 from the publisher metadata, never from T12.

A download requires Mendeley Data API access. If its API returns an access
error, download via the official website and run --inspect-local instead.
This tool never treats an unrelated schema fixture as external validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET_ID = 'zhj366m53p'
VERSION = 2
DOI = '10.17632/zhj366m53p.2'
OFFICIAL = f'https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}'
API = f'https://api.data.mendeley.com/datasets/publics/{DATASET_ID}/files?version={VERSION}'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(2**20), b''):
            h.update(buf)
    return h.hexdigest()


def parse_catalog(data: object) -> list[dict]:
    if isinstance(data, dict):
        data = data.get('results', data.get('files', []))
    if not isinstance(data, list):
        raise ValueError('Publisher catalog response is not an array of files')
    result = []
    for x in data:
        if not isinstance(x, dict):
            continue
        filename = x.get('filename') or x.get('name') or ''
        if str(filename).lower().endswith('.csv'):
            details = x.get('content_details') or {}
            result.append({'filename': filename, 'file_id': x.get('id'),
                           'size': x.get('size') or details.get('size'),
                           'source_sha256': details.get('sha256_hash'),
                           'download_url': details.get('download_url')})
    return result


def load_catalog() -> list[dict]:
    req = urllib.request.Request(API, headers={'Accept':'application/json',
                                               'User-Agent':'T12-academic-research/1.0'})
    with urllib.request.urlopen(req, timeout=40) as response:
        return parse_catalog(json.load(response))


def provenance(path: Path, metadata: dict | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    info = {'dataset':'MoMTSim', 'doi': DOI, 'version': VERSION,
            'publisher_page':OFFICIAL,'filename':path.name,
            'bytes':path.stat().st_size, 'sha256':sha256(path),
            'source':'Mendeley Data v2 — user/publisher supplied file',
            'not_t12_generated_data':True,
            'originality_attested_by_researcher':False}
    if metadata:
        info['publisher_catalog_file'] = metadata
        if metadata.get('source_sha256'):
            info['matches_publisher_sha256'] = info['sha256'].lower() == str(metadata['source_sha256']).lower()
            if not info['matches_publisher_sha256']:
                raise ValueError('SHA256 mismatch against publisher metadata')
    return info


def download_file(info: dict, dest_dir: Path) -> Path:
    url = info.get('download_url')
    if not url:
        raise RuntimeError('Publisher did not expose a download URL; download manually from '+OFFICIAL)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != 'https':
        raise ValueError('Refusing non-HTTPS dataset URL')
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(info['filename']).name
    if dest.exists():
        raise FileExistsError(f'{dest} exists; use --inspect-local instead')
    part = dest.with_suffix(dest.suffix+'.part')
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'T12/1.0'}), timeout=90) as response, part.open('wb') as output:
            for b in iter(lambda: response.read(2**20), b''):
                output.write(b)
        if info.get('size') and part.stat().st_size != int(info['size']):
            raise ValueError('Download size mismatch')
        part.rename(dest)
        provenance(dest, info)
        return dest
    finally:
        if part.exists():
            part.unlink()


def main():
    p=argparse.ArgumentParser(description='Publisher-origin MoMTSim v2 acquisition / hash inventory')
    p.add_argument('--catalog',action='store_true',help='Inspect original publisher metadata')
    p.add_argument('--download',action='store_true',help='Download one official CSV; first prefer 1000 dataset')
    p.add_argument('--inspect-local',type=Path,help='Compute SHA256 of a manually downloaded official file')
    p.add_argument('--output',type=Path,default=Path('data/external'))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.inspect_local:
        meta=provenance(a.inspect_local)
        out=a.output/'momtsim_provenance.json'
        out.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(meta,indent=2,ensure_ascii=False)); return
    try:
        catalog=load_catalog()
    except (OSError, ValueError, urllib.error.HTTPError) as e:
        raise SystemExit(f'Mendeley API unavailable ({e}). Download manually at {OFFICIAL}; then use --inspect-local. No original dataset was validated.')
    if not catalog:
        raise SystemExit('Mendeley returned no CSV file metadata; inspect the official web page manually.')
    print(json.dumps(catalog,indent=2,ensure_ascii=False))
    if a.download:
        files=sorted(catalog,key=lambda x:('1000_dataset' not in x['filename'], x['filename']))
        local=download_file(files[0],a.output)
        meta=provenance(local,files[0])
        (a.output/'momtsim_provenance.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
        print('Downloaded:',local,'SHA256:',meta['sha256'])

if __name__=='__main__':main()
