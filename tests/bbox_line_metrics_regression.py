"""Adversarial detector metric checks, no OCR or network."""
from bbox_line_metrics import grade, evaluate
import copy
import hashlib
import json
from pathlib import Path
import tempfile


def run():
    a, b = [.1,.1,.9,.2], [.1,.3,.9,.4]
    lines = [{'rect':r, 'words':[{'rect':r}]} for r in (a,b)]
    exact = grade(lines, [a,b])
    assert exact['clean_recall'] == exact['clean_precision'] == 1
    assert exact['clean_word_area_coverage'] == [1,1]
    giant = grade(lines, [[0,0,1,1]])
    assert giant['clean_recall'] == 0 and giant['merged_candidates'] == 1
    assert giant['matched_lines'] == 1
    split = grade(lines, [[.1,.1,.5,.2],[.5,.1,.9,.2],b])
    assert split['split_lines'] == 1 and split['clean_recall'] == .5
    assert grade(lines, [])['missed_lines'] == 2
    partial = grade(lines, [[.1,.13,.9,.2],b])
    assert partial['clean_recall'] == 1
    assert abs(partial['clean_word_area_coverage'][0]-.7) < 1e-8
    extra = grade(lines, [a,b,[.1,.6,.9,.7]])
    assert extra['spurious_lines'] == 1 and extra['clean_precision'] == 2/3
    for bad in ([0,0,float('nan'),1],[0,0,0,1],[False,0,1,1]):
        try: grade(lines,[bad])
        except ValueError: pass
        else: raise AssertionError('invalid rectangle accepted')
    # Reports must reject held-out tuning, image drift and changed review
    # geometry even when the altered rectangle would improve the score.
    from bbox_annotation_workflow_regression import fixture, reviewed_for
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        manifest, _, _ = fixture()
        dev = next(p for p in manifest['pages'] if p['split'] == 'dev')
        image = root/dev['image']['path']
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b'synthetic image provenance')
        dev['image']['sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
        reviewed = reviewed_for(dev)
        mp, dp, rp = [root/name for name in ('manifest.json','detector.json','reviewed.json')]
        mp.write_text(json.dumps(manifest))
        detector = {'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
                    'detector':'synthetic', 'version':'test', 'pages':[
                        {'page_id':dev['page_id'], 'image_sha256':dev['image']['sha256'], 'rects':[a]}]}
        dp.write_text(json.dumps(detector)); rp.write_text(json.dumps([reviewed]))
        assert evaluate(mp,dp,rp)['pages'][0]['clean_recall'] == 1
        changed = copy.deepcopy(reviewed)
        changed['draft']['lines'][0]['rect'] = [0,0,1,1]
        rp.write_text(json.dumps([changed]))
        try: evaluate(mp,dp,rp)
        except ValueError: pass
        else: raise AssertionError('tampered review accepted')
        rp.write_text(json.dumps([reviewed])); image.write_bytes(b'different image')
        try: evaluate(mp,dp,rp)
        except ValueError: pass
        else: raise AssertionError('changed image accepted')
        dev['split'] = 'heldout'; mp.write_text(json.dumps(manifest))
        detector['manifest_sha256'] = hashlib.sha256(mp.read_bytes()).hexdigest()
        dp.write_text(json.dumps(detector))
        try: evaluate(mp,dp,rp)
        except ValueError as exc: assert 'development' in str(exc)
        else: raise AssertionError('held-out tuning accepted')
    print('LINE METRIC REGRESSIONS PASSED')


if __name__ == '__main__':
    run()
