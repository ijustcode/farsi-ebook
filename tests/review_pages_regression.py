"""Explicit review scope must never enqueue or mutate unselected pages."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from click.testing import CliRunner
from farsi2epub import review
from farsi2epub.cli import main


def run():
    ws = SimpleNamespace(slug='synthetic', meta={'page_count':200},
                         pages_done=lambda:[61,128,129], review_dir=Path('/tmp/review-pages-test'))
    with patch.object(review, '_read_sidecar', side_effect=AssertionError('read outside scope')), \
         patch.object(review, '_write_sidecar', side_effect=AssertionError('mutated sidecar')):
        assert review._select_pages_for_review(ws, pages=[128,61]) == ([61,128], [])
        try: review._select_pages_for_review(ws, pages=[62])
        except ValueError: pass
        else: raise AssertionError('untranscribed selection accepted')
    runner=CliRunner()
    with patch('farsi2epub.cli._load_workspace', return_value=ws), \
         patch.object(review,'read_server_state',return_value=None), \
         patch.object(review,'run_review') as start, \
         patch.object(review,'launch_review_background',return_value='http://example.invalid') as background:
        for flags in ([],['--_child'],['--background']):
            result=runner.invoke(main,['review','synthetic','--pages','61,128','--wait-for-boxes',*flags])
            assert result.exit_code==0,result.output
            call=background if '--background' in flags else start
            assert call.call_args.kwargs['pages']==[61,128]
            assert call.call_args.kwargs['wait_for_boxes'] is True
        for spec in ('62','xyz','300'):
            result=runner.invoke(main,['review','synthetic','--pages',spec])
            assert result.exit_code==2,result.output
    with patch('farsi2epub.cli._load_workspace',return_value=ws), \
         patch.object(review,'read_server_state',return_value={'url':'http://existing','pid':1}), \
         patch.object(review,'run_review') as start:
        result=runner.invoke(main,['review','synthetic','--pages','61,128'])
        assert result.exit_code==1 and '--stop' in result.output
        start.assert_not_called()
    # A terminal unresolved page is finished; pending pages must be polled again.
    with patch.object(review, '_boxes_payload', side_effect=[
        {'pending': True}, {'pending': False}, {'pending': False}
    ]) as payload, patch.object(review.time, 'sleep') as sleep:
        review._wait_for_boxes(ws, [61, 128], SimpleNamespace(is_alive=lambda: True))
        assert payload.call_count == 3
        sleep.assert_called_once_with(0.5)
    with patch.object(review, '_boxes_payload') as payload:
        review._wait_for_boxes(ws, [61], SimpleNamespace(is_alive=lambda: False))
        payload.assert_not_called()
    print('REVIEW PAGE SCOPE REGRESSIONS PASSED')


if __name__=='__main__':
    run()
