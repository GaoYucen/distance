"""Acquire or verify the exact public source bytes used by R2.
Run from an audit branch containing reports/audit-r2-20260913/data_audit.json.
Only published GitHub and Figshare files are accessed; no cloud-folder scraping.
Existing mismatched files are never overwritten.
"""
from pathlib import Path
import argparse, hashlib, json
from urllib.parse import urlparse
import requests

ROOT=Path(__file__).resolve().parents[1]


def check_or_get(session,base,relative,url,size,sha256,verify_only):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe relative path')
    if urlparse(url).scheme!='https' or urlparse(url).hostname not in ('raw.githubusercontent.com','ndownloader.figshare.com'):
        raise ValueError('Unapproved source host')
    if size>50_000_000:raise ValueError('Unexpected source size')
    target=base/rel
    if target.exists():raw=target.read_bytes()
    elif verify_only:raise FileNotFoundError(target)
    else:
        response=session.get(url,timeout=90);response.raise_for_status();raw=response.content
    if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=sha256:
        raise ValueError(f'Source size/hash mismatch: {relative}')
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.part')
        temporary.write_bytes(raw);temporary.replace(target)
    print('SOURCE_VERIFIED',relative,len(raw),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    report=json.loads((ROOT/'reports/audit-r2-20260913/data_audit.json').read_text())
    upstream=report['survey_manifest'];native=report['native_manifest']
    assert upstream['upstream_repo']=='purduedb/shortest-distance-survey'
    assert upstream['upstream_commit']=='dcaa89d38300bfb823eda84ccdfc85c42edbeae8'
    session=requests.Session();session.headers['User-Agent']='distance-reproducibility-audit/2'
    base=ROOT/'data/survey_dcaa89d'
    for f in upstream['files']:
        url=f"https://raw.githubusercontent.com/{upstream['upstream_repo']}/{upstream['upstream_commit']}/{f['path']}"
        check_or_get(session,base,f['path'],url,f['bytes'],f['sha256'],args.verify_only)
    raw=ROOT/'data/figshare_native_20260913'
    for f in native['files']:
        check_or_get(session,raw,f['name'],f['download_url'],f['bytes'],f['sha256'],args.verify_only)
    if not args.verify_only:
        for target,value in [(base/'manifest.json',upstream),(raw/'manifest.json',native)]:
            if not target.exists():target.write_text(json.dumps(value,indent=2)+'\n')
    print('ALL_R2_SOURCES_VERIFIED',len(upstream['files'])+len(native['files']),flush=True)

if __name__=='__main__':main()
