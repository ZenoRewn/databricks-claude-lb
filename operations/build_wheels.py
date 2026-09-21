"""Download exact public PyPI wheels with SHA-256 verification. Author: Zeno Ren."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.tags import cpython_tags,compatible_tags,parse_tag
from packaging.utils import parse_wheel_filename


def download(lock,output,python_version,architecture):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    version=tuple(map(int,python_version.split('.')))
    platform_arch='x86_64' if architecture=='amd64' else 'aarch64'
    platforms=[f'manylinux_2_{n}_{platform_arch}' for n in range(39,16,-1)]+[f'manylinux2014_{platform_arch}',f'linux_{platform_arch}']
    tags=list(cpython_tags(version,abis=['cp'+''.join(map(str,version))],platforms=platforms))+list(compatible_tags(version,platforms=platforms))
    order={tag:i for i,tag in enumerate(tags)}
    requirements=[Requirement(line) for line in Path(lock).read_text().splitlines() if line.strip() and not line.startswith('#')]
    def fetch(requirement):
        specs=list(requirement.specifier)
        if len(specs)!=1 or specs[0].operator!='==':raise ValueError('Wheel input must be exactly pinned')
        v=specs[0].version;name=requirement.name
        with tempfile.TemporaryDirectory() as directory:
            metadata=Path(directory)/'metadata.json'
            subprocess.run(['curl','--fail','--silent','--show-error','--retry','3','--retry-all-errors','--connect-timeout','10','--max-time','60',
                            '-o',str(metadata),f'https://pypi.org/pypi/{name}/{v}/json'],check=True,stderr=subprocess.PIPE)
            candidates=[]
            for item in json.loads(metadata.read_text())['urls']:
                if item['packagetype']!='bdist_wheel':continue
                _,_,_,wheel_tags=parse_wheel_filename(item['filename'])
                ranks=[order[t] for t in wheel_tags if t in order]
                if ranks:candidates.append((min(ranks),item))
            if not candidates:raise ValueError('No compatible wheel: '+name+'=='+v)
            cached=[pair for pair in candidates if (output/pair[1]['filename']).is_file()
                    and hashlib.sha256((output/pair[1]['filename']).read_bytes()).hexdigest()==pair[1]['digests']['sha256']]
            item=min(cached or candidates,key=lambda pair:pair[0])[1]
            parsed=urlsplit(item['url'])
            if parsed.scheme!='https' or parsed.hostname!='files.pythonhosted.org':raise ValueError('Unexpected wheel host')
            expected=item['digests']['sha256'];target=output/item['filename']
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                temporary=Path(directory)/'wheel'
                subprocess.run(['curl','--fail','--silent','--show-error','--retry','3','--retry-all-errors','--connect-timeout','10','--max-time','120',
                                '-o',str(temporary),item['url']],check=True,stderr=subprocess.PIPE)
                if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected:raise ValueError('Wheel digest mismatch')
                target.write_bytes(temporary.read_bytes())
            return {'name':name,'version':v,'file':target.name,'sha256':expected}
    with ThreadPoolExecutor(max_workers=4) as workers:result=list(workers.map(fetch,requirements))
    (output/f'manifest-{python_version}-{architecture}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--python',default='3.12');parser.add_argument('--architecture',choices=('amd64','arm64'),default='amd64')
    args=parser.parse_args()
    try:
        items=download(args.lock,args.output,args.python,args.architecture)
        print(json.dumps({'verified_wheels':len(items),'python':args.python,'architecture':args.architecture}))
    except Exception as exc:
        details={'error_type':type(exc).__name__}
        if isinstance(exc,subprocess.CalledProcessError):
            details.update(returncode=exc.returncode,host=urlsplit(exc.cmd[-1]).hostname,
                           resource=urlsplit(exc.cmd[-1]).path.rsplit('/',1)[-1],
                           diagnostic=(exc.stderr or b'').decode(errors='replace')[:200])
        print(json.dumps(details));raise SystemExit(1)
