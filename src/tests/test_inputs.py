import json
import pytest
from colbert_ppi.input_validation import read_inference_pairs


def test_generated_pair_ids_are_distinct(tmp_path):
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{'left_tokens': [0, 5, 2], 'right_tokens': [0, 6, 2]}] * 2))
    assert [r['pair_id'] for r in read_inference_pairs(path)] == ['pair_1', 'pair_2']


@pytest.mark.parametrize('tokens', [[], [0, 2], [0, 1, 2], [0, -1, 2], [0, 4.5, 2], [0, True, 2], [4, 5, 2]])
def test_malformed_tokens_fail_before_encoding(tmp_path, tokens):
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{'left_tokens': tokens, 'right_tokens': [0, 6, 2]}]))
    with pytest.raises(ValueError):
        read_inference_pairs(path)
