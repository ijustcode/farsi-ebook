"""Development-only line detector evaluation against independently reviewed ink.

Edges require at least half of a truth line's width and height to overlap.
Clean recall requires an isolated one-to-one edge, so a page-sized rectangle
or many fragments cannot win by merely intersecting every printed line.
These metrics help select a detector; they never constitute box promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

VERSION = 1


def rect(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or not 0 <= x <= 1 for x in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        raise ValueError("invalid normalized rectangle")
    return value


def overlap(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])), max(0., min(a[3], b[3])-max(a[1], b[1]))


def grade(lines, candidates):
    truth = [rect(line['rect']) for line in lines]
    boxes = [rect(box) for box in candidates]
    edges = []
    for t in truth:
        edges.append([j for j, b in enumerate(boxes)
                      if overlap(t, b)[0] >= .5*(t[2]-t[0])
                      and overlap(t, b)[1] >= .5*(t[3]-t[1])])
    reverse = [[i for i, js in enumerate(edges) if j in js] for j in range(len(boxes))]
    clean = [(i, js[0]) for i, js in enumerate(edges)
             if len(js) == 1 and len(reverse[js[0]]) == 1]
    # Maximum bipartite matching is diagnostic; clean recall penalizes merges
    # and splits explicitly instead of hiding them in arbitrary assignments.
    matched = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False
    for i in range(len(truth)):
        augment(i, set())
    coverage = []
    for i, j in clean:
        words = lines[i]['words']
        areas, covered = [], []
        for word in words:
            w = rect(word['rect'])
            areas.append((w[2]-w[0])*(w[3]-w[1]))
            x, y = overlap(w, boxes[j])
            covered.append(x*y)
        coverage.append(sum(covered)/sum(areas) if areas else None)
    return {'truth_lines': len(truth), 'detected_lines': len(boxes),
            'matched_lines': len(matched), 'clean_lines': len(clean),
            'clean_recall': len(clean)/len(truth) if truth else None,
            'clean_precision': len(clean)/len(boxes) if boxes else None,
            'missed_lines': sum(not js for js in edges),
            'spurious_lines': sum(not ids for ids in reverse),
            'split_lines': sum(len(js) > 1 for js in edges),
            'merged_candidates': sum(len(ids) > 1 for ids in reverse),
            'clean_pairs': [list(pair) for pair in clean],
            'clean_word_area_coverage': coverage}


def evaluate(manifest_path, detector_path, reviews_path):
    from bbox_annotation_workflow import validate_reviewed_page
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    detector = json.loads(detector_path.read_bytes())
    if detector['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError('detector manifest drift')
    pages = {p['page_id']: p for p in manifest['pages']}
    detected = {p['page_id']: p for p in detector['pages']}
    if len(detected) != len(detector['pages']):
        raise ValueError('duplicate detector page')
    reviewed = json.loads(reviews_path.read_bytes())
    results, seen = [], set()
    for item in reviewed:
        packet, draft, review = item['packet'], item['draft'], item['review']
        errors = validate_reviewed_page(packet, draft, review)
        if errors:
            raise ValueError('; '.join(errors))
        pid = packet['page_id']
        if pid in seen:
            raise ValueError('duplicate reviewed page')
        seen.add(pid)
        if pid not in pages or pages[pid]['split'] != 'dev':
            raise ValueError('detector selection only accepts development pages')
        page = pages[pid]
        if pid not in detected:
            raise ValueError('missing detector page; refusing survivor subset')
        if packet['image_sha256'] != page['image']['sha256'] or detected[pid]['image_sha256'] != packet['image_sha256']:
            raise ValueError('image provenance drift')
        image = manifest_path.parent/page['image']['path']
        if hashlib.sha256(image.read_bytes()).hexdigest() != packet['image_sha256']:
            raise ValueError('image bytes changed')
        results.append({'page_id':pid, 'image_sha256':packet['image_sha256'],
                        **grade(draft['lines'], detected[pid]['rects'])})
    if not results:
        raise ValueError('no independently reviewed development pages')
    return {'instrument_version':VERSION, 'detector':detector['detector'],
            'detector_version':detector['version'],
            'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
            'reviews_sha256':hashlib.sha256(reviews_path.read_bytes()).hexdigest(),
            'detector_report_sha256':hashlib.sha256(detector_path.read_bytes()).hexdigest(),
            'promotion_eligible':False, 'pages':results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--detector-report', type=Path, required=True)
    p.add_argument('--reviews', type=Path, required=True,
                   help='JSON array of {packet,draft,review}; independent approved development pages only')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = evaluate(args.manifest, args.detector_report, args.reviews)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(f"Scored {len(report['pages'])} reviewed development pages; no promotion decision")


if __name__ == '__main__':
    main()
