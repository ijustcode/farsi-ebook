"""Create a self-contained image comparison from the frozen experiment."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

import fitz


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('out/levenshtein_fragment_final'))
    parser.add_argument('--visual', type=Path)
    args = parser.parse_args()
    report = json.loads((args.out/'comparison.json').read_text())
    inputs = json.loads((args.out/'inputs.json').read_text())
    assert report['inputs_sha256'] == sha(args.out/'inputs.json'), 'input drift'
    pages = {p['page']: p for p in inputs['pages']}
    cases = []
    for row in report['corrections']:
        if not row['changed']:
            continue
        n, q = row['page'], row['query']
        path = args.out/'images'/f'{n:04d}.png'
        assert sha(path) == pages[n]['image_sha256'], 'image drift'
        image = fitz.Pixmap(fitz.csGRAY, fitz.Pixmap(path))
        w, h = image.width, image.height
        rects = [s for name in ('baseline','candidate') for s in row[name]['box']['segments']]
        x0 = max(0, math.floor((min(s['x0'] for s in rects)-.12)*w))
        x1 = min(w, math.ceil((max(s['x1'] for s in rects)+.12)*w))
        y0 = max(0, math.floor((min(s['y0'] for s in rects)-.035)*h))
        y1 = min(h, math.ceil((max(s['y1'] for s in rects)+.035)*h))
        bounds = fitz.IRect(x0,y0,x1,y1)
        crop = fitz.Pixmap(fitz.csGRAY, bounds, False)
        crop.copy(image, bounds)
        md = pages[n]['md']
        span = q['span']
        context = md[max(0,span[0]-120):min(len(md),span[1]+120)] if span else ''
        item = {'page': n, 'key': row['key'], 'query': q['text'], 'context': context,
                'suggestion': '\n'.join(q['alts']),
                'width': x1-x0, 'height': y1-y0,
                'image': 'data:image/jpeg;base64,'+base64.b64encode(crop.tobytes('jpeg', jpg_quality=88)).decode()}
        for name in ('baseline','candidate'):
            item[name] = [[s['x0']*w-x0,s['y0']*h-y0,s['x1']*w-x0,s['y1']*h-y0]
                          for s in row[name]['box']['segments']]
        cases.append(item)
    data = {'comparison_sha256': sha(args.out/'comparison.json'),
            'inputs_sha256': report['inputs_sha256'], 'cases': cases}
    template = (Path(__file__).parent/'data'/'fragment_review.html').read_text()
    fragment = template.replace('REVIEW_DATA', json.dumps(data,ensure_ascii=False).replace('<','\\u003c'))
    target = args.visual or args.out/'placement-review.html'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(fragment)
    assert target.stat().st_size < 1_000_000, 'review exceeds inline size limit'
    print(f'{len(cases)} changed-box comparisons: {target.resolve()} ({target.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
