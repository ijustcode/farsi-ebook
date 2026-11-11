"""Transcription review gate on scanned pages (plan item T3, Step 0).

The plan read hadpi's stored `needs_review: false` as a gate failure. It is
not: on a page with no embedded-text oracle the gate already flags a page whose
self-reported confidence is below validators.CONF_REVIEW (0.70) or that carries
the `illegible` flag. The stored false was written later by the review UI's
Accept (review.py sets needs_review=False and reviewed="edited"). This script
pins the gate so that cannot silently regress. Offline; no API, no books/
needed (the real-sidecar check skips when the workspace is absent).
"""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farsi2epub import llm, transcribe, validators
from farsi2epub.config import MODEL_STRONG, RES_HI, RES_STD

# Values copied from books/hadpi-articles/text/000N.json (scanned, no oracle).
HADPI = [(1, .45, ['illegible']), (2, .55, ['illegible']), (3, .60, ['marginalia']),
         (4, .65, []), (5, .68, ['marginalia'])]
NO_ORACLE = {'issues': [], 'embedded_similarity': None, 'length_ratio': None,
             'latin_ratio': 0.0, 'arabic_codepoints': 0}
TEXT = 'ماده ۱ این شرکت سهامی خاص است و نام آن شرکت هادپی است.'


def _result(conf, flags):
    return llm.PageTranscription(is_blank=False, text_md=TEXT, headings=[], flags=flags,
                                 starts_mid_paragraph=False, continues_next=False, confidence=conf)


def _run_page(conf, flags, model, resolution, embedded=None):
    calls = []
    def fake_transcribe(client, png, model_, n, page_count, title_hint, extra_hint=None):
        calls.append((model_, png))
        return _result(conf, flags), {'input_tokens': 1, 'output_tokens': 1}, 0.0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ws = SimpleNamespace(slug='synthetic', meta={'title_fa': None, 'source_type': 'scanned'},
                             pdf_path=root/'source.pdf',
                             page_md_path=lambda n: root/f'{n:04d}.md',
                             page_meta_path=lambda n: root/f'{n:04d}.json')
        with patch.object(llm, 'transcribe_page', side_effect=fake_transcribe), \
             patch.object(transcribe, '_page_image', side_effect=lambda ws_, n, res: res.encode()), \
             patch.object(transcribe, 'extract_embedded_text', return_value=embedded), \
             patch.object(transcribe, 'extract_heading_candidates', return_value=[]), \
             patch.object(transcribe.qc, 'record_event', side_effect=AssertionError('history write')):
            transcribe._transcribe_one(ws, None, 1, 5, model, resolution, transcribe.RunState())
        return json.loads((root/'0001.json').read_text()), calls


def run():
    assert validators.CONF_REVIEW == 0.70 and 'illegible' in validators.REVIEW_FLAGS
    # 1. hadpi values: every page is flagged by the gate itself.
    for page, conf, flags in HADPI:
        checks = validators.evaluate(TEXT, conf, flags, None)
        assert checks['embedded_similarity'] is None and checks['issues'] == []
        assert validators.needs_review(conf, flags, checks, False), page
        # Already Sonnet + hi-res: there is no stronger rung to escalate to.
        sidecar, calls = _run_page(conf, flags, MODEL_STRONG, RES_HI)
        assert sidecar['needs_review'] is True and sidecar['escalated'] is False, (page, sidecar)
        assert len(calls) == 1
        # The economy path does escalate the same page to Sonnet + hi-res.
        sidecar, calls = _run_page(conf, flags, MODEL_STRONG, RES_STD)
        assert sidecar['escalated'] is True and sidecar['needs_review'] is True
        assert [c[0] for c in calls] == [MODEL_STRONG, MODEL_STRONG] and calls[1][1] == RES_HI.encode()
    # The illegible flag alone flags a no-oracle page, whatever the confidence.
    assert validators.needs_review(.95, ['illegible'], NO_ORACLE, False)
    # 2. Typical clean scanned pages are not flagged (medians: bachehaye_ghali
    # 0.75, ghessehaye-majid 0.90, jameeshenasi_khodemani 0.90).
    for conf, flags in ((.75, ['marginalia']), (.90, []), (.85, ['image']), (.70, [])):
        assert not validators.needs_review(conf, flags, validators.evaluate(TEXT, conf, flags, None), False)
        sidecar, _ = _run_page(conf, flags, MODEL_STRONG, RES_HI)
        assert sidecar['needs_review'] is False
    # 3. Digital pages with a working oracle: outcome decided by similarity, as before.
    words = ' '.join(f'واژه{i}' for i in range(40))
    good = validators.evaluate(words, .92, [], words)
    assert good['embedded_similarity'] == 1.0 and not validators.needs_review(.92, [], good, False)
    bad = validators.evaluate(words, .92, [], ' '.join(f'دیگر{i}' for i in range(40)))
    assert bad['embedded_similarity'] < validators.EMBED_SIM_REVIEW
    assert validators.needs_review(.92, [], bad, False)
    assert not validators.needs_review(.2, ['illegible'], NO_ORACLE, True)  # blank pages never
    # 4. Optional: the real hadpi sidecars (skip cleanly if the workspace is absent).
    text_dir = Path(__file__).resolve().parent.parent/'books'/'hadpi-articles'/'text'
    if text_dir.is_dir():
        for page, _conf, _flags in HADPI:
            sidecar = json.loads((text_dir/f'{page:04d}.json').read_text())
            assert validators.needs_review(sidecar['confidence'], sidecar['flags'], sidecar['validators'],
                                           sidecar['is_blank']), page
            # The stored false is the post-review state, not the gate's output.
            assert sidecar['needs_review'] is False and sidecar.get('reviewed') in {'edited', 'accepted'}
        print('real hadpi sidecars: gate flags 5/5; stored false is post-review')
    else:
        print('real hadpi sidecars: SKIPPED (workspace absent)')
    print('REVIEW GATE REGRESSIONS PASSED')


if __name__ == '__main__':
    run()
