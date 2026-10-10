"""Checks for fair controls, topic isolation, and patient-level reporting."""
import json
import random
from types import SimpleNamespace

import pytest

from trials_domain.negative_selector import TrialsNegativeSelector
from trials_domain.patient_metrics import patient_report
from trials_domain.soft_sampling import distribution, sample, prepare_text_scores
from scripts import run_mesh_soft_sampling as study


class Matcher:
    def ontology_set_similarity(self, a, b):
        return float(bool(set(a) & set(b)))


def selector(guidance):
    hard, easy = [f'h{i}' for i in range(20)], [f'e{i}' for i in range(40)]
    views = {n: {'nct_id': n, 'encoder_view': n,
                 'coarse_uris': ['D1'] if n.endswith('0') else ['D2']}
             for n in hard + easy}
    result = TrialsNegativeSelector(
        {'q': {'ineligible': hard, 'not_relevant': easy}}, views,
        matcher=Matcher() if guidance in ('mesh', 'hybrid') else None,
        soft_guidance=guidance, soft_mix=.5, mesh_facet='disease',
        training_seed=42, total_epochs=15)
    result.text_scores = {'q': {n: float(n.endswith('1')) for n in easy}}
    anchor = SimpleNamespace(metadata={'resume_id': 'q'},
                             resume={'coarse_uris': ['D1']},
                             job={'nct_id': 'positive'})
    return result, anchor


def selected(selector, anchor, epoch):
    values, _ = selector.select_batch_negatives(anchor, [], 7, epoch)
    return [(v['nct_id'], v['grade']) for v in values]


@pytest.mark.parametrize('scores', [[None]*4, [0]*4, [0, None, 0, None]])
def test_no_information_is_uniform(scores):
    assert distribution(scores, .2) == [.25]*4


def test_ties_missing_values_and_full_pool_support():
    probs = distribution([1., 1., 0., None], .2)
    assert probs[0] == probs[1]
    assert probs[0] > probs[3] > probs[2] > 0
    pool = [str(i) for i in range(30)]
    guidance = {n: 100. if n=='0' else 0. for n in pool}
    draws = [sample(pool, 7, random.Random(i), [guidance], .5, .2)
             for i in range(100)]
    assert all(len(row)==len(set(row))==7 for row in draws)
    assert set().union(*map(set, draws)) == set(pool)


def test_no_signal_sampling_matches_uniform_and_ignores_file_order():
    pool = [str(i) for i in range(15)]
    a = sample(pool, 7, random.Random(42), [], .5, .2)
    b = sample(pool[::-1], 7, random.Random(42), [{n:0 for n in pool}], .5, .2)
    assert a == b


@pytest.mark.parametrize('guidance', study.ARMS)
@pytest.mark.parametrize('epoch,hard_count', [(0,1), (14,4)])
def test_grade_mix_and_ineligible_draws_are_identical(guidance, epoch, hard_count):
    control, anchor = selector('uniform')
    candidate, sample_anchor = selector(guidance)
    baseline = selected(control, anchor, epoch)
    values = selected(candidate, sample_anchor, epoch)
    assert len(values)==7
    assert sum(g==1 for _,g in values)==hard_count
    assert {n for n,g in values if g==1} == {n for n,g in baseline if g==1}
    assert values == selected(candidate, sample_anchor, epoch)


@pytest.mark.parametrize('guidance', study.ARMS)
def test_validation_candidates_identical_and_do_not_need_mining_scores(guidance):
    control, anchor = selector('uniform')
    candidate, sample_anchor = selector(guidance)
    candidate.common_validation_topics = {'q'}
    candidate.text_scores = {}
    assert selected(candidate, sample_anchor, 14)==selected(control, anchor, 14)


def test_distinct_positives_get_variety_within_an_epoch():
    candidate, anchor = selector('uniform')
    a = selected(candidate, anchor, 0)
    anchor.job['nct_id'] = 'another_positive'
    assert a != selected(candidate, anchor, 0)


def test_metrics_do_not_compare_candidates_across_patients():
    # Within each patient ranking is perfect, despite cross-patient offsets.
    report = patient_report(['a','a','b','b'], [.2,.1,.9,.8], [2,0,2,0])
    assert report['macro']['eligible_auc']==1.
    assert report['macro']['eligible_map']==1.
    assert 'eligible_vs_ineligible' not in report['n_topics']
    assert report['ndcg_gains']=={'0':0,'1':1,'2':3}
    assert patient_report([None,None], [.2,.1], [2,0]) is None
    with pytest.raises(ValueError):
        patient_report(['a',None], [.2,.1], [2,0])


def test_folds_are_disjoint_and_exclude_2022(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    source = tmp_path/'preprocess/trec_ct'
    source.mkdir(parents=True)
    topics = [{'topic_id':f'{year}_{i}', 'year':str(year),
               'encoder_view':f'patient {year} {i}', 'condition_signal_present':True,
               'mesh_uris':['D1'], 'condition_uris':['D1']}
              for year, count in ((2021,6),(2022,2)) for i in range(count)]
    trials = [{'nct_id':f'n{grade}', 'title':f'trial {grade}',
               'encoder_view':f'trial {grade}', 'mesh_uris':['D1'],
               'condition_uris':['D1']} for grade in range(3)]
    judgments = [{'topic_id':t['topic_id'],'nct_id':f'n{g}','grade':g}
                 for t in topics for g in range(3)]
    for name, rows in [('topics',topics),('trials',trials),('qrels',judgments)]:
        study.write_rows(source/f'{name}.jsonl', rows)
    splits = study.prepare_folds(tmp_path/'study', 3)
    held_out=[]
    for train, positive_val, split in splits:
        training=study.read_rows(train)
        validation=study.read_rows(split/'validation.jsonl')
        a={r['resume']['topic_id'] for r in training}
        b={r['resume']['topic_id'] for r in validation}
        assert not a & b
        assert len(a)==4 and len(b)==2
        assert all(t.startswith('2021_') for t in a|b)
        assert all(r['job']['grade']==2 for r in training+study.read_rows(positive_val))
        assert {r['job']['grade'] for r in validation}=={0,1,2}
        assert {r['topic_id'] for r in study.read_rows(split/'negative_pools.jsonl') if r['split']=='validation'}==b
        held_out.extend(b)
    assert len(held_out)==len(set(held_out))==6


def test_summary_averages_seeds_within_patient():
    results=[]
    for seed, delta in [(13,.1),(42,.3)]:
        for arm in ('uniform','mesh'):
            value=.5+(delta if arm=='mesh' else 0)
            results.append({'arm':arm,'fold':0,'seed':seed,'pooled_auc':value,
                            'per_patient':{'macro':{'eligible_map':value},
                                           'per_topic':{'2021_1':{'eligible_map':value}}}})
    summary=study.summarize(results)['arms']['mesh']
    assert summary['patients']==1
    assert summary['patient_mean_delta']==pytest.approx(.2)
    assert summary['patient_bootstrap_ci95']==pytest.approx([.2,.2])


def test_text_mining_excludes_validation_and_test_topics(tmp_path):
    import torch
    candidate, _ = selector('text')
    candidate.pools={'train':{'not_relevant':['e1']}, 'validation':{'not_relevant':['e2']}}
    candidate.common_validation_topics={'validation'}
    candidate.text_scores={}
    study.write_rows(tmp_path/'topics.jsonl', [
        {'topic_id':t,'encoder_view':t} for t in ('train','validation','2022_1')])
    calls=[]
    def encode(view, role):
        calls.append(view['encoder_view'])
        return torch.tensor([1.,0.])
    trainer=SimpleNamespace(freeze_text_encoder=True,
                            embedding_cache=SimpleNamespace(cache={},get_content_key=lambda v:v['encoder_view']),
                            _encode_content_to_text_embedding=encode)
    assert prepare_text_scores(trainer,candidate,tmp_path)==1
    assert set(candidate.text_scores)=={'train'}
    assert set(calls)=={'train','e1'}


def test_study_configs_have_shared_validation_and_no_ontology_weighting():
    configs=[study.config_for(a) for a in study.ARMS]
    for cfg in configs:
        assert cfg['trials_common_validation'] is True
        assert cfg['ontology_weight']==0 and cfg['loss_type']=='infonce'
    differing={k for k in configs[0] if len({json.dumps(c[k],sort_keys=True) for c in configs})>1}
    assert differing=={'trials_soft_guidance','trials_mesh_tiered_negatives'}
