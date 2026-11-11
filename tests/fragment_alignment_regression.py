"""Adversarial contracts for the opt-in fragment experiment (no API/book)."""
from dataclasses import asdict
from pathlib import Path
import sys
import random
import hashlib
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farsi2epub.fragment_alignment import closest_substring, infer_words, normalize
from farsi2epub.page_map import PrintedWord
from farsi2epub.scan import PlacementService
from fragment_alignment_experiment import Collector


def observation(text, rect, key='crop', line=0):
    return {'text': text, 'rect': rect, 'key': key, 'line': line}


def run():
    assert normalize('كی\u200cها ۱۲٣') == normalize('کیها 123')
    assert normalize('abc') != normalize('cba')
    a, reason = closest_substring('defgh', 'abcdefghi')
    assert reason == 'accepted' and (a.start, a.end, a.distance) == (3, 8, 0)
    for fragment in ('abcxefgh', 'abcdfgh', 'abcdxefgh'):
        a, reason = closest_substring(fragment, 'abcdefgh')
        assert reason == 'accepted' and a.distance == 1, (fragment, a, reason)
    a, _ = closest_substring('abcdfgh', 'abcdefgh')
    assert 4 not in a.matched  # an omitted line character is not observed
    a, _ = closest_substring('می\u200cرود خانه', 'میرودخانه')
    assert a.distance == 0
    for fragment, line, expected in [
            ('abcd', 'abcdxxxxabcd', 'ambiguous_occurrence'),
            ('abcd', 'zzzzzzzz', 'edit_distance'),
            ('ab', 'abcd', 'short_or_unreadable'),
            ('abcd', 'abc�def', 'short_or_unreadable'),
            ('a'*513, 'a'*520, 'input_too_long')]:
        a, reason = closest_substring(fragment, line)
        assert a is None and reason == expected, (fragment, reason)
    # Independent full-string edit-distance oracle across every substring.
    # Accepted semiglobal matches must attain the actual closest distance.
    def distance(a, b):
        prev = list(range(len(b)+1))
        for i, ca in enumerate(a, 1):
            row = [i]
            for j, cb in enumerate(b, 1):
                row.append(min(row[-1]+1, prev[j]+1, prev[j-1]+(ca != cb)))
            prev = row
        return prev[-1]
    rng = random.Random(27)
    for _ in range(100):
        fragment = ''.join(rng.choice('abcd') for _ in range(5))
        line = ''.join(rng.choice('abcd') for _ in range(12))
        a, _ = closest_substring(fragment, line)
        if a:
            assert a.distance == min(distance(fragment, line[i:j])
                                     for i in range(len(line)) for j in range(i+1,len(line)+1))

    words = [PrintedWord('abcdefgh', [.45,.1,.95,.2], 0, ['line'], False),
             PrintedWord('ijklmnop', [.05,.1,.5,.2], 0, ['line'], False)]
    original = [asdict(w) for w in words]
    obs = [observation('abcdef', [.7,.1,.9,.2]),
           observation('ijklmnop', [.1,.1,.3,.2], 'second')]
    mapped, changed, diag = infer_words(words, obs, [])
    assert changed == {0,1} and all(not w.supported for w in mapped)
    assert mapped[0].rect == obs[0]['rect']
    assert mapped[1].rect == obs[1]['rect']
    assert [asdict(w) for w in words] == original
    fuzzy = [observation('abcxefgh', [.7,.1,.9,.2])]
    assert infer_words(words, fuzzy, [])[1] == {0}
    assert not infer_words(words, fuzzy, [], allow_edits=False)[1]
    # Too little of the word leaves its baseline geometry intact.
    mapped, changed, _ = infer_words(words, [observation('abcd', [.7,.1,.9,.2])], [])
    assert not changed and mapped == words
    # Several partial readings can cover one word, including split/fused tokens.
    mapped, changed, _ = infer_words(words, [observation('abcd', [.8,.1,.9,.2]),
                                            observation('efgh', [.6,.1,.7,.2], 'tail')], [])
    assert changed == {0} and all(abs(a-b) < 1e-6 for a,b in zip(mapped[0].rect, [.6,.1,.9,.2]))
    mapped, changed, _ = infer_words(words, [observation('abcdefgh ijklmnop', [.2,.1,.8,.2])], [])
    assert changed == {0,1} and mapped[0].rect == mapped[1].rect
    # Physical reversal contradicts the line reading; reject both, don't pick one.
    reversed_obs = [observation('abcdefgh', [.1,.1,.3,.2]),
                    observation('ijklmnop', [.7,.1,.9,.2], 'second')]
    mapped, changed, diag = infer_words(words, reversed_obs, [])
    assert not changed and all(d['reason'] == 'neighbor_geometry_conflict' for d in diag)
    # Supported words/anchors are immutable and fence off impossible locations.
    words[0].supported = True
    words[0].rect = [.65,.1,.75,.2]
    mapped, changed, diag = infer_words(words, [observation('ijklmnop', [.8,.1,.95,.2])], [])
    assert mapped[0] == words[0] and not changed
    assert diag[0]['reason'] == 'supported_anchor_conflict'
    # No raw reading, no inferred textual evidence. Anonymous ink is not OCR.
    mapped, changed, _ = infer_words(words, [], [])
    assert not changed and mapped == words
    anonymous = [PrintedWord('�', [.1,.1,.9,.2], 0, [], False)]
    assert not infer_words(anonymous, obs, [])[1]
    # Observations from another line cannot shift a target across lines.
    assert not infer_words(words, [observation('ijklmnop', [.1,.1,.3,.2], line=1)], [])[1]
    # A shared raw-image key must retain multiple physical occurrences. This
    # checks extraction bookkeeping, not just repeated-string matching.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        collector = Collector(SimpleNamespace(root=root,pdf_path=root/'source.pdf'), root/'scratch')
        png = b'identical rendered crop bytes'
        key = collector._observation_key(png)
        path = collector.source_root/'readings'/f'{key}.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'key':key,'complete':True,'legible':True,
                                   'lines':['abcdefgh'],'image_sha256':hashlib.sha256(png).hexdigest()}))
        with patch.object(PlacementService,'_render',return_value=png):
            page = SimpleNamespace(rect=fitz.Rect(0,0,100,100))
            first = collector._render(page,fitz.Rect(10,10,30,20))
            second = collector._render(page,fitz.Rect(60,10,80,20))
            collector._read([first,second],1,False)
        obs = list(collector.observations.values())
        assert len(obs) == 2 and obs[0]['rect'] != obs[1]['rect']
        assert obs[0]['key'] == obs[1]['key'] and collector.budget.spent == 0
        collector.close()
    print('FRAGMENT ALIGNMENT REGRESSIONS PASSED')


if __name__ == '__main__':
    run()
