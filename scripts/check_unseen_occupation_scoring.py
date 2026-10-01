"""Check that the saved shared text projection can score new ESCO profiles.

This verifies execution and finite scores, not zero-shot accuracy or transfer.
"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from sentence_transformers import SentenceTransformer
from run_phase1_embedding_evaluation import CareerAwareContrastiveModel, content_to_text

def check():
    torch.set_num_threads(4)
    training_path = ROOT / 'preprocess/learning_curve_v7/frac_100/train.jsonl'
    with training_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    observed = {r['job'].get('occupation_uri') for r in rows}
    with (ROOT / 'dataset/esco/occupations_en.csv').open() as f:
        candidates = [r for r in csv.DictReader(f) if r['conceptUri'] not in observed][:2]
    checkpoint_path = ROOT / 'results/results_v7_vanilla/phase1_pretraining/best_checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['model_state_dict']
    input_dim = state['projection_head.0.weight'].shape[1]
    output_dim = state['projection_head.3.weight'].shape[0]
    model = CareerAwareContrastiveModel(input_dim=input_dim, projection_dim=output_dim)
    model.load_state_dict(state)
    model.eval()
    pilot_config = json.loads((ROOT / 'config/trials_criterion_pilot.json').read_text())
    if not Path(pilot_config['text_model_path']).exists():
        with torch.inference_mode():
            projected = model(torch.zeros(3, input_dim))
        return {'status': 'checkpoint_and_projection_verified; text_encoding_unavailable',
                'checkpoint': str(checkpoint_path.relative_to(ROOT)),
                'checkpoint_loaded': True, 'input_dimension': input_dim,
                'projection_dimension': output_dim,
                'projection_output_finite': bool(torch.isfinite(projected).all()),
                'projection_check_inputs': 'zero vectors; shape/execution check only',
                'new_text_scoring_completed': False,
                'candidates_absent_from_canonical_v7_training': [
                    {'occupation_uri': r['conceptUri'], 'name': r['preferredLabel']} for r in candidates],
                'limitation': 'Pinned local MPNet model directory is missing; new text could not be encoded. No zero-shot accuracy claim.'}
    encoder = SentenceTransformer(pilot_config['text_model_path'], device='cpu', local_files_only=True)
    query = content_to_text(rows[0]['resume'], 'resume')
    profiles = [content_to_text({'title': r['preferredLabel'], 'description': r.get('description', '')}, 'job')
                for r in candidates]
    with torch.inference_mode():
        encoded = encoder.encode([query] + profiles, convert_to_tensor=True, show_progress_bar=False)
        projected = model(encoded)
        scores = projected[1:] @ projected[0]
    assert torch.isfinite(scores).all()
    return {'status': 'finite_scores_for_new_occupation_profiles',
            'checkpoint': str(checkpoint_path.relative_to(ROOT)),
            'candidate_absence_checked_against': str(training_path.relative_to(ROOT)),
            'input_dimension': input_dim, 'projection_dimension': output_dim,
            'candidate_count': len(candidates),
            'candidates': [{'occupation_uri': r['conceptUri'], 'name': r['preferredLabel'], 'score': float(s)}
                           for r, s in zip(candidates, scores)],
            'limitations': ['Execution check only; no ground-truth relevance labels for these profiles.',
                            'Candidate absence checked against canonical v7 training file, not a reconstructed historical checkpoint manifest.',
                            'Frozen pretrained language model may already know these occupations; this is interaction-level novelty.']}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    result = check()
    out = ROOT / 'results/occupation_task_readiness/unseen_scoring_check.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
