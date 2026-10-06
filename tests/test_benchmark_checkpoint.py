import asyncio
import copy
import json
import multiprocessing
import os
from pathlib import Path
import signal

import pandas as pd
import pytest

import benchmark
import benchmark_checkpoint as bc
import cli
from dataset_builder import compute_pair_id
from policy import PairOutput


def source(n=3):
    return pd.DataFrame([dict(contribution_id='1', contributionContent='Source', comment_text=f'Comment {i}',
        key_message=target, key_message_type='proposal', related_sentences=[target],
        parent_context=dict(available=True, parent_kind='comment', parent_comment_text='Parent'))
        for i in range(n) for target in ('Target a', 'Target b')])


def preparation(tmp_path, n=3):
    frame = source(n)
    path = tmp_path / 'source.parquet'
    frame.to_parquet(path)
    ref = benchmark.expected_cohort(frame).assign(resolution_status='resolved', reference_label='neutral')
    ref_path = tmp_path / 'reference.parquet'
    ref.to_parquet(ref_path)
    return dict(source=frame, source_path=path, reference_path=ref_path, folds_path=None,
        output_path=tmp_path / 'run.parquet', model_name='mock', base_url='http://proxy.invalid', verify_ionos=False)


def rows(cp, index, status='accepted'):
    return [{**identity, 'status': status, 'proposed_stance': 'neutral' if status == 'accepted' else None,
        'final_stance': 'neutral' if status == 'accepted' else None,
        'basis': 'unaddressed' if status == 'accepted' else None, 'comment_evidence': [], 'source_evidence': [],
        'explanation': '', 'review_reasons': [] if status == 'accepted' else ['test_reason']}
        for identity in cp.identities[index]]


def run(cp, **overrides):
    args = dict(source=cp.preparation['source'], base_url=cp.manifest['endpoint'], model_name='mock',
        verify_ionos=False, run_id=cp.manifest['run_id'], checkpoint=cp)
    args.update(overrides)
    return asyncio.run(benchmark.run_benchmark(**args))


def fake_engine(monkeypatch, call):
    class Engine:
        def __init__(self, **kwargs):
            pass

        async def resolve_comment(self, client, semaphore, context):
            call(context)
            return [PairOutput(pair_id=compute_pair_id(context.contribution_id, km.text, context.comment_text),
                key_message_id=km.key_message_id, run_id=context.run_id, proposed_stance='neutral',
                final_stance='neutral', status='accepted', basis='unaddressed') for km in context.key_messages]

    monkeypatch.setattr(benchmark, 'StancePolicyEngineV2', Engine)
    return Engine


def test_all_statuses_are_completed_and_skipped(tmp_path, monkeypatch):
    args = preparation(tmp_path, 4)
    with bc.BenchmarkCheckpoint(**args) as cp:
        original = cp.manifest['run_id']
        for i, status in enumerate(('accepted', 'needs_review', 'failed')):
            cp.commit_comment(rows(cp, i, status))
    calls = []
    fake_engine(monkeypatch, lambda ctx: calls.append(ctx.comment_text))
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        assert cp.manifest['run_id'] == original
        assert cp.completed_comments == 3
        assert len(run(cp, run_id=None)) == 8
        cp.publish()
    assert calls == ['Comment 3']
    monkeypatch.setattr(benchmark, 'verify_ionos_only', lambda *a: pytest.fail('no network for finalization'))
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        assert cp.state['published']
        assert len(run(cp)) == 8
        cp.publish()
    assert args['output_path'].exists()


def test_checkpoint_32_of_40_partial_missing_denominator(tmp_path):
    import evaluator_v2
    args = preparation(tmp_path, 40)
    with bc.BenchmarkCheckpoint(**args) as cp:
        for i in range(32):
            cp.commit_comment(rows(cp, i))
        predictions = cp.export_partial()
        metrics = evaluator_v2.evaluate(predictions, pd.read_parquet(args['reference_path']), expected_pairs=cp.expected)
        assert metrics.eligible_pairs == 80 and metrics.missing_prediction_pairs == 16
        assert metrics.forced_choice_accuracy == .8
        with pytest.raises(ValueError, match='complete coverage'):
            cp.publish()
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        assert cp.completed_comments == 32 and len(cp.rows()) == 64


@pytest.mark.parametrize('change', ['source', 'reference', 'folds', 'parent', 'order', 'model', 'endpoint',
                                  'verification', 'retry', 'code', 'runtime'])
def test_contract_changes_fail_read_only_before_network(tmp_path, monkeypatch, change):
    args = preparation(tmp_path)
    with bc.BenchmarkCheckpoint(**args) as cp:
        cp.commit_comment(rows(cp, 0))
    before = {p: p.read_bytes() for p in tmp_path.glob('run.*')}
    if change in ('source', 'reference'):
        path = args[change + '_path']
        path.write_bytes(path.read_bytes() + b'changed')
    elif change == 'folds':
        path = tmp_path / 'folds.json'
        path.write_text('{}')
        args['folds_path'] = path
    elif change == 'parent':
        args['source'].at[0, 'parent_context'] = dict(available=False)
        args['source'].at[1, 'parent_context'] = dict(available=False)
    elif change == 'order':
        args['source'] = args['source'].iloc[::-1]
    elif change == 'model':
        args['model_name'] = 'other'
    elif change == 'endpoint':
        args['base_url'] = 'http://other.invalid'
    elif change == 'verification':
        args['verify_ionos'] = True
    elif change == 'retry':
        args['execution_settings'] = {**bc.DEFAULT_EXECUTION_SETTINGS, 'max_retries': 0}
    else:
        original = benchmark.prepare_benchmark

        def changed(**kwargs):
            expected, manifest = original(**kwargs)
            manifest['runtime' if change == 'runtime' else 'code_state']['changed'] = True
            return expected, manifest

        monkeypatch.setattr(benchmark, 'prepare_benchmark', changed)
    monkeypatch.setattr(benchmark, 'verify_ionos_only', lambda *a: pytest.fail('network'))
    with pytest.raises(ValueError):
        with bc.BenchmarkCheckpoint(**args, resume=True):
            pytest.fail('contract accepted')
    assert before == {p: p.read_bytes() for p in before}


def rewrite(path, mutate):
    envelope = bc.strict_load(path)
    mutate(envelope['payload'])
    envelope['checksum'] = bc.fingerprint(envelope['payload'])
    path.write_bytes(bc.canonical(envelope))


@pytest.mark.parametrize('corruption', ['truncated', 'duplicate_key', 'nan', 'overflow', 'checksum', 'version',
    'bool_count', 'duplicate_comment', 'wrong_order', 'missing_target', 'duplicate_target', 'mixed_run',
    'mixed_model', 'mixed_source', 'status', 'bad_list', 'mixed_target', 'wrong_count', 'extra_column'])
def test_corrupt_checkpoints_rejected_without_repair(tmp_path, corruption):
    args = preparation(tmp_path)
    with bc.BenchmarkCheckpoint(**args) as cp:
        cp.commit_comment(rows(cp, 0))
    path = args['output_path'].with_suffix('.checkpoint.json')
    if corruption == 'truncated':
        path.write_bytes(path.read_bytes()[:80])
    elif corruption == 'duplicate_key':
        path.write_bytes(b'{"version":1,' + path.read_bytes()[1:])
    elif corruption in ('nan', 'overflow'):
        path.write_bytes(path.read_bytes().replace(b'"version":1', b'"version":' + (b'NaN' if corruption == 'nan' else b'1e999')))
    elif corruption in ('checksum', 'version'):
        value = bc.strict_load(path)
        value[corruption] = 'bad' if corruption == 'checksum' else True
        path.write_bytes(bc.canonical(value))
    else:
        def mutate(s):
            record = s['records'][0]
            row = record['rows'][0]
            if corruption == 'bool_count': s['completed_comments'] = True
            elif corruption == 'duplicate_comment':
                s['records'].append(copy.deepcopy(record))
                s['completed_comments'] += 1
                s['completed_pairs'] += 2
            elif corruption == 'wrong_order': record['ordinal'] = 1
            elif corruption == 'missing_target': record['rows'].pop()
            elif corruption == 'duplicate_target': record['rows'][1] = copy.deepcopy(row)
            elif corruption == 'mixed_run': row['run_id'] = 'benchmark_other'
            elif corruption == 'mixed_model': row['stance_model_name'] = 'other'
            elif corruption == 'mixed_source': row['comment_text'] = 'other'
            elif corruption == 'mixed_target': row['key_message_id'] = 'other'
            elif corruption == 'status': row['status'] = 'not_a_status'
            elif corruption == 'bad_list': row['comment_evidence'] = 'not a list'
            elif corruption == 'wrong_count': s['completed_pairs'] = 1
            elif corruption == 'extra_column': row['extra'] = 1
        rewrite(path, mutate)
    original = path.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        with bc.BenchmarkCheckpoint(**args, resume=True):
            pytest.fail('corruption accepted')
    assert path.read_bytes() == original


@pytest.mark.parametrize('suffix', ['.partial.parquet', '.parquet'])
def test_unrelated_output_never_clobbered(tmp_path, suffix):
    args = preparation(tmp_path)
    with bc.BenchmarkCheckpoint(**args):
        pass
    output = args['output_path'].with_suffix(suffix)
    pd.DataFrame({'unrelated': [1]}).to_parquet(output)
    before = output.read_bytes()
    with pytest.raises(ValueError, match='Unrelated'):
        with bc.BenchmarkCheckpoint(**args, resume=True):
            pass
    assert output.read_bytes() == before


def test_stale_owned_partial_regenerated(tmp_path):
    args = preparation(tmp_path)
    with bc.BenchmarkCheckpoint(**args) as cp:
        cp.commit_comment(rows(cp, 0))
        cp.export_partial()
        cp.commit_comment(rows(cp, 1))
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        assert len(pd.read_parquet(cp.partial)) == 2
        cp.export_partial()
        assert len(pd.read_parquet(cp.partial)) == 4


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'reordered', 'run', 'target'])
def test_exact_result_coverage_checked_before_zip(tmp_path, monkeypatch, bad):
    args = preparation(tmp_path)
    engine = fake_engine(monkeypatch, lambda ctx: None)
    original = engine.resolve_comment

    async def broken(self, *a):
        results = await original(self, *a)
        if bad == 'missing': results.pop()
        elif bad == 'duplicate': results[1] = results[0]
        elif bad == 'reordered': results.reverse()
        elif bad == 'run': results[0].run_id = 'other'
        elif bad == 'target': results[0].key_message_id = 'other'
        return results

    monkeypatch.setattr(engine, 'resolve_comment', broken)
    with pytest.raises(ValueError, match='coverage'):
        with bc.BenchmarkCheckpoint(**args) as cp:
            run(cp)
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        assert cp.completed_comments == 0


def test_persistence_failure_stops_before_next_inference(tmp_path, monkeypatch):
    args = preparation(tmp_path)
    calls = []
    fake_engine(monkeypatch, lambda ctx: calls.append(ctx.comment_text))
    original = bc.atomic_json

    def fail(path, data):
        if Path(path).suffix == '.json' and data.get('payload', {}).get('completed_comments') == 1:
            raise OSError('disk full')
        original(path, data)

    with pytest.raises(bc.CheckpointFailure):
        with bc.BenchmarkCheckpoint(**args) as cp:
            monkeypatch.setattr(bc, 'atomic_json', fail)
            run(cp)
    assert calls == ['Comment 0']
    assert bc.strict_load(args['output_path'].with_suffix('.checkpoint.json'))['payload']['completed_comments'] == 0
    monkeypatch.setattr(bc, 'atomic_json', original)
    with bc.BenchmarkCheckpoint(**args, resume=True, recover_stale_lock=True) as cp:
        assert cp.completed_comments == 0


@pytest.mark.parametrize('point', ['file_fsync', 'replace', 'dir_fsync'])
def test_atomic_failure_is_old_or_new_never_torn(tmp_path, monkeypatch, point):
    path = tmp_path / 'atomic.json'
    bc.atomic_json(path, {'old': True})
    original_fsync = bc.os.fsync
    count = 0

    def fsync(fd):
        nonlocal count
        count += 1
        if count == (1 if point == 'file_fsync' else 2):
            raise OSError('injected')
        original_fsync(fd)

    if point == 'replace':
        monkeypatch.setattr(bc.os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('injected')))
    else:
        monkeypatch.setattr(bc.os, 'fsync', fsync)
    with pytest.raises(OSError):
        bc.atomic_json(path, {'new': True})
    assert bc.strict_load(path) == ({'new': True} if point == 'dir_fsync' else {'old': True})
    assert not list(tmp_path.glob('*.tmp'))


def child_run(args, connection, point):
    with bc.BenchmarkCheckpoint(**args) as cp:
        cp.commit_comment(rows(cp, 0))
        if point == 'during_next_call':
            class Engine:
                def __init__(self, **kwargs): pass
                async def resolve_comment(self, *a):
                    connection.send('ready')
                    await asyncio.Event().wait()
            benchmark.StancePolicyEngineV2 = Engine
            run(cp)
        else:
            if point == 'after_parquet':
                cp._export(cp.output)
            connection.send('ready')
            signal.pause()


@pytest.mark.parametrize('point', ['after_commit', 'during_next_call', 'after_parquet'])
def test_real_lock_kill_recovery_and_finalization(tmp_path, monkeypatch, point):
    args = preparation(tmp_path, 1 if point == 'after_parquet' else 2)
    ctx = multiprocessing.get_context('spawn')
    receive, send = ctx.Pipe(duplex=False)
    child = ctx.Process(target=child_run, args=(args, send, point))
    child.start()
    try:
        assert receive.poll(30) and receive.recv() == 'ready'
        inode = args['output_path'].with_suffix('.lock').stat().st_ino
        with pytest.raises(ValueError, match='live process'):
            with bc.BenchmarkCheckpoint(**args, resume=True, recover_stale_lock=True):
                pass
        os.kill(child.pid, signal.SIGKILL)
        child.join(10)
        assert child.exitcode == -signal.SIGKILL
        with pytest.raises(ValueError, match='Stale session'):
            with bc.BenchmarkCheckpoint(**args, resume=True):
                pass
        calls = []
        fake_engine(monkeypatch, lambda context: calls.append(context.comment_text))
        with bc.BenchmarkCheckpoint(**args, resume=True, recover_stale_lock=True) as cp:
            assert cp.completed_comments == 1
            run(cp)
            cp.publish()
        assert calls == ([] if point == 'after_parquet' else ['Comment 1'])
        assert args['output_path'].with_suffix('.lock').stat().st_ino == inode
    finally:
        if child.is_alive():
            child.kill()
        child.join(10)
        receive.close()
        send.close()


@pytest.mark.parametrize('flags', [['--dry-run', '--resume'], ['--recover-stale-lock'], ['--evaluate-partial']])
def test_invalid_cli_combinations_before_io(flags):
    with pytest.raises(ValueError):
        cli.main(['benchmark', '--source', 'absent', '--output', 'absent', *flags])


@pytest.mark.parametrize('endpoint', ['http://user:pass@proxy', 'http://proxy?token=secret', 'http://proxy#secret'])
def test_hidden_endpoint_components_rejected(tmp_path, endpoint):
    args = preparation(tmp_path)
    args['base_url'] = endpoint
    with pytest.raises(ValueError, match='Endpoint'):
        with bc.BenchmarkCheckpoint(**args):
            pass


def test_history_without_checkpoint_cannot_resume(tmp_path):
    args = preparation(tmp_path)
    args['output_path'].write_bytes(b'old')
    with pytest.raises(ValueError, match='historical'):
        with bc.BenchmarkCheckpoint(**args, resume=True):
            pass


def test_final_export_failure_then_publish_without_inference(tmp_path, monkeypatch):
    args = preparation(tmp_path, 1)
    with pytest.raises(OSError):
        with bc.BenchmarkCheckpoint(**args) as cp:
            cp.commit_comment(rows(cp, 0))
            monkeypatch.setattr(cp, '_export', lambda *a: (_ for _ in ()).throw(OSError('injected')))
            cp.publish()
    monkeypatch.setattr(benchmark, 'StancePolicyEngineV2', lambda **kw: pytest.fail('no engine'))
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        run(cp)
        cp.publish()


def test_route_changes_block_resume_before_inference(tmp_path, monkeypatch):
    from ionos_config import ModelRouteInfo
    args = preparation(tmp_path)
    args['verify_ionos'] = True
    with bc.BenchmarkCheckpoint(**args) as cp:
        cp.record_session({'run_id': cp.manifest['run_id'], 'endpoint': args['base_url'],
            'verification_requested': True, 'provider_verified': True, 'verification_status': 'verified',
            'route': {'upstream_model': 'original', 'api_base': 'https://openai.inference.de-txl.ionos.com/v1', 'is_ionos': True}})
    monkeypatch.setattr(benchmark, 'verify_ionos_only', lambda *a: ModelRouteInfo('mock', 'changed',
        'https://openai.inference.de-txl.ionos.com/v1', True))
    monkeypatch.setattr(benchmark, 'StancePolicyEngineV2', lambda **kw: pytest.fail('no engine'))
    with pytest.raises(bc.CheckpointFailure):
        with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
            run(cp, verify_ionos=True)


def test_cli_partial_is_explicit_no_network_and_full_denominator(tmp_path, monkeypatch, capsys):
    args = preparation(tmp_path, 2)
    args.update(folds_path='', mode='two_stage', limit=None, enable_thinking=False,
                verifier_enabled=False, role='primary', tiers=['A', 'B'], revisable_variant='baseline')
    with bc.BenchmarkCheckpoint(**args) as cp:
        cp.commit_comment(rows(cp, 0))
    monkeypatch.setattr(cli, 'load_dotenv', lambda: None)
    monkeypatch.setattr(cli, 'run_benchmark', lambda **kw: pytest.fail('no inference'))
    cli.main(['benchmark', '--source', str(args['source_path']), '--reference', str(args['reference_path']),
              '--output', str(args['output_path']), '--folds', '', '--model-name', 'mock',
              '--base-url', args['base_url'], '--no-verify-ionos', '--resume', '--evaluate-partial'])
    output = capsys.readouterr().out
    assert 'missing_prediction_pairs=2' in output and 'forced_choice_accuracy=0.500 (2/4)' in output
    assert not args['output_path'].exists()


def test_credential_rotation_and_provider_verification_each_session(tmp_path, monkeypatch):
    from ionos_config import ModelRouteInfo
    args = preparation(tmp_path, 2)
    args['verify_ionos'] = True
    keys = []

    def verify(model, endpoint, key):
        keys.append(key)
        return ModelRouteInfo(model, 'same', 'https://openai.inference.de-txl.ionos.com/v1', True)

    monkeypatch.setattr(benchmark, 'verify_ionos_only', verify)
    calls = []

    def interrupt(ctx):
        calls.append(ctx.comment_text)
        if len(calls) == 2:
            raise KeyboardInterrupt

    fake_engine(monkeypatch, interrupt)
    with pytest.raises(KeyboardInterrupt):
        with bc.BenchmarkCheckpoint(**args) as cp:
            run(cp, verify_ionos=True, api_key='secret-one')
    fake_engine(monkeypatch, lambda ctx: None)
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        run(cp, verify_ionos=True, api_key='secret-two')
        cp.publish()
        assert len(cp.state['sessions']) == 2
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        run(cp, verify_ionos=True, api_key='secret-three')
    assert keys == ['secret-one', 'secret-two']
    serialized = args['output_path'].with_suffix('.checkpoint.json').read_text()
    assert 'secret-' not in serialized


@pytest.mark.parametrize('variant', ['baseline', 'source_bound_evidence_ids'])
def test_real_revisable_outputs_commit_interrupt_resume(tmp_path, monkeypatch, variant):
    from unittest.mock import AsyncMock
    from context_builder import build_comment_context
    import revisable_attribution as ra

    cases = [
        ('explicit_exclusion', label, None) for label in ('neutral', 'against', 'in_favor')
    ] + [
        (basis, label, uncertainty)
        for uncertainty in ('target', 'scope')
        for basis, label in [('explicit_target', 'against'), ('explicit_target', 'in_favor'),
                             ('explicit_exclusion', 'neutral'), ('unaddressed', None),
                             ('no_position', None)]
    ] + [('unresolved', label, None) for label in (None, 'in_favor', 'against', 'neutral')]
    cases += [('schema', None, None), ('failed', None, None), ('unaddressed', None, None)]
    args = preparation(tmp_path, len(cases))
    args.update(mode='revisable_attribution', revisable_variant=variant)
    replies = []
    for entry, (basis, label, uncertainty) in zip(benchmark.load_frozen_comments(args['source']), cases, strict=True):
        ctx = build_comment_context(run_id='wire', **entry)
        scope = {'scope_type': 'whole_contribution_approval', 'comment_evidence': [ctx.comment_text]}
        targets = [dict(key_message_id=km.key_message_id, attribution=basis, stance=label,
                        comment_evidence=[ctx.comment_text], source_evidence=km.related_sentences,
                        explanation='Preserve explanation',
                        uncertainty_flags=['suspected_sarcasm'] if uncertainty == 'target' else [])
                   for km in ctx.key_messages]
        second = {'revised_scope': copy.deepcopy(scope), 'targets': targets}
        if uncertainty == 'scope':
            second['revised_scope']['uncertainty_flags'] = ['suspected_sarcasm']
        if variant == 'source_bound_evidence_ids':
            table = ra.build_evidence_table(ctx)
            for record in (scope, second['revised_scope'], *targets):
                record['comment_evidence_ids'] = [e['id'] for e in table if e['id'].startswith('c:')]
                record.pop('comment_evidence')
                if 'source_evidence' in record:
                    record['source_evidence_ids'] = [e['id'] for e in table
                        if e['target_binding'] == record['key_message_id']]
                    record.pop('source_evidence')
        replies.extend([None] if basis == 'failed' else [scope, {} if basis == 'schema' else second])
    handler = AsyncMock()
    handler.call_model.side_effect = replies
    original_commit = bc.BenchmarkCheckpoint.commit_comment
    prefix = []

    def interrupt(cp, values):
        original_commit(cp, values)
        if cp.completed_comments == len(cases) - 1:
            prefix.extend(copy.deepcopy(cp.rows()))
            raise KeyboardInterrupt

    monkeypatch.setattr(bc.BenchmarkCheckpoint, 'commit_comment', interrupt)
    with pytest.raises(KeyboardInterrupt):
        with bc.BenchmarkCheckpoint(**args) as cp:
            run(cp, mode='revisable_attribution', revisable_variant=variant, llm_handler=handler)
    assert {row['status'] for row in prefix} == {'accepted', 'needs_review', 'failed'}
    assert [row['basis'] for row in prefix[:6]] == ['explicit_exclusion'] * 6
    for i, (basis, label, uncertainty) in enumerate(cases[:-1]):
        row = prefix[2 * i]
        if basis in ('schema', 'failed'):
            assert row['basis'] is row['proposed_stance'] is row['final_stance'] is None
        elif uncertainty or basis == 'unresolved':
            assert row['status'] == 'needs_review' and row['final_stance'] is None
            assert row['proposed_stance'] == ('in_favor' if basis in ('unaddressed', 'no_position') else label)
            assert row['basis'] == ('general_approval' if basis in ('unaddressed', 'no_position') else basis)
        else:
            assert row['status'] == 'accepted' and row['final_stance'] == label
        if basis not in ('schema', 'failed'):
            assert row['comment_evidence'] == [f'Comment {i}']
            assert row['source_evidence'] == ['Target a']
            assert row['explanation']
    import pyarrow.parquet as pq
    assert pq.read_table(args['output_path'].with_suffix('.partial.parquet')).to_pylist() == prefix
    monkeypatch.setattr(bc.BenchmarkCheckpoint, 'commit_comment', original_commit)
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        assert cp.rows() == prefix
        calls = handler.call_model.await_count
        run(cp, mode='revisable_attribution', revisable_variant=variant, llm_handler=handler)
        assert handler.call_model.await_count == calls + 2
        assert cp.rows()[:-2] == prefix
        saved = copy.deepcopy(cp.rows())
        cp.publish()
    assert pq.read_table(args['output_path']).to_pylist() == saved
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        run(cp, mode='revisable_attribution', revisable_variant=variant, llm_handler=handler)
        assert cp.rows() == saved
    assert handler.call_model.await_count == len(replies)


@pytest.mark.parametrize('status,bases', [
    ('accepted', ['general_approval', 'explicit_target', 'explicit_exclusion', 'unaddressed', 'no_position']),
    ('needs_review', ['general_approval', 'explicit_target', 'explicit_exclusion', 'unaddressed', 'no_position', 'unresolved', None]),
    ('failed', [None]),
])
def test_row_contract_string_lists_and_empty_failure_reasons(tmp_path, status, bases):
    args = preparation(tmp_path, len(bases))
    with bc.BenchmarkCheckpoint(**args) as cp:
        for i, basis in enumerate(bases):
            values = rows(cp, i, status)
            for row in values:
                row.update(basis=basis, review_reasons=[], comment_evidence=[' exact\r\nquote '],
                           source_evidence=[' source '], explanation=' explanation ')
            if status == 'needs_review':
                values[0].update(proposed_stance='neutral', review_reasons=['', 'arbitrary reason'])
            cp.commit_comment(values)
        expected = copy.deepcopy(cp.rows())
        cp.publish()
    with bc.BenchmarkCheckpoint(**args, resume=True) as cp:
        assert cp.rows() == expected


@pytest.mark.parametrize('invalid', [
    {'status': 'unknown'}, {'basis': 'unknown'}, {'proposed_stance': 'unknown'},
    {'final_stance': 'unknown'}, {'basis': None}, {'basis': 'unresolved'},
    {'proposed_stance': None, 'final_stance': None}, {'proposed_stance': 'against'},
    {'review_reasons': ['not empty']}, {'status': 'needs_review'},
    {'status': 'needs_review', 'final_stance': None, 'basis': 'unknown'},
    {'status': 'failed', 'final_stance': None, 'basis': None},
    {'status': 'failed', 'final_stance': None, 'proposed_stance': None},
    {'status': 'needs_review', 'final_stance': None, 'review_reasons': [None]},
    {'comment_evidence': [1]}, {'source_evidence': 'quote'}, {'explanation': None},
])
@pytest.mark.parametrize('persisted', [False, True])
def test_invalid_row_metadata_rejected_before_mutation_or_provider(tmp_path, monkeypatch, invalid, persisted):
    args = preparation(tmp_path)
    monkeypatch.setattr(benchmark, 'verify_ionos_only', lambda *a: pytest.fail('provider called'))
    with bc.BenchmarkCheckpoint(**args) as cp:
        if persisted:
            cp.commit_comment(rows(cp, 0))
        else:
            values = rows(cp, 0)
            values[0].update(invalid)
            before = bc.canonical(cp.state)
            artifacts = {p: p.read_bytes() for p in tmp_path.glob('run.*')}
            with pytest.raises(bc.RowValidationError) as error:
                cp.commit_comment(values)
            assert error.value.code == 'row_validation_failed'
            assert bc.canonical(cp.state) == before
            assert all(p.read_bytes() == data for p, data in artifacts.items())
    if persisted:
        rewrite(cp.path, lambda state: state['records'][0]['rows'][0].update(invalid))
        artifacts = {p: p.read_bytes() for p in tmp_path.glob('run.*')}
        with pytest.raises(bc.RowValidationError):
            with bc.BenchmarkCheckpoint(**args, resume=True) as resumed:
                run(resumed)
        assert all(p.read_bytes() == data for p, data in artifacts.items())
