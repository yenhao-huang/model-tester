#!/usr/bin/env python3
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

OUT_ROOT = Path('/Users/yenhaohuang/Desktop/datasets/tts')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Focus on commonly used TTS/ASR speech datasets on HF.
DATASETS = [
    'keithito/lj_speech',
    'openslr/librispeech_asr',
    'mozilla-foundation/common_voice_17_0',
    'google/fleurs',
]

api = HfApi()


def safe_download(repo_id: str, filename: str, out_dir: Path):
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=filename,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
        return {'file': filename, 'status': 'ok', 'local': local}
    except Exception as e:
        return {'file': filename, 'status': 'error', 'error': str(e)}


def main():
    report = []

    for repo_id in DATASETS:
        target = OUT_ROOT / repo_id.replace('/', '__')
        target.mkdir(parents=True, exist_ok=True)

        item = {'dataset': repo_id, 'status': 'ok', 'downloads': []}
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')
            (target / 'FILELIST.txt').write_text('\n'.join(files), encoding='utf-8')

            preferred = [f for f in files if f.lower().endswith('readme.md')]
            preferred += [f for f in files if f.lower().endswith('.json')][:3]
            preferred += [f for f in files if f.lower().endswith('.txt')][:3]

            seen = set()
            selected = []
            for f in preferred:
                if f not in seen:
                    selected.append(f)
                    seen.add(f)
                if len(selected) >= 6:
                    break

            if not selected:
                selected = files[:3]

            for f in selected:
                item['downloads'].append(safe_download(repo_id, f, target))

        except Exception as e:
            item['status'] = 'error'
            item['error'] = str(e)

        report.append(item)

    (OUT_ROOT / 'download_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
