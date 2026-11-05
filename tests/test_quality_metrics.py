# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest
from looplab.quality_metrics import holm_adjust, numeric_answer, paired_comparison, sample_scores


def scores(values):
    return {('task', i): {'correct': value, 'doc_hash': str(i), 'target_hash': str(i)}
            for i, value in enumerate(values)}


def test_paired_outcomes_and_ties():
    result = paired_comparison(scores([False] * 8), scores([True] * 8), bootstrap=200)
    assert result['wins'] == 8 and result['losses'] == 0
    assert result['two_sided_p'] == 2 / 256
    assert result['paired_bootstrap_95_percent_interval'] == [100, 100]
    tied = paired_comparison(scores([True, False]), scores([True, False]), bootstrap=200)
    assert tied['two_sided_p'] == 1 and tied['accuracy_change_points'] == 0


def test_pairing_rejects_different_questions_and_answers():
    with pytest.raises(ValueError, match='identical question'):
        paired_comparison(scores([True]), scores([True, False]))
    changed = scores([True])
    changed[('task', 0)]['target_hash'] = 'changed'
    with pytest.raises(ValueError, match='target changed'):
        paired_comparison(scores([True]), changed)


def test_multiple_comparisons_are_adjusted_without_reordering():
    assert holm_adjust([.05, .001, .02]) == pytest.approx([.05, .003, .04])
    assert holm_adjust([1, 1]) == [1, 1]


def test_sample_filter_is_explicit_and_duplicate_ids_fail():
    row = {'doc_id': 3, 'filter': 'flexible-extract', 'exact_match': 1, 'doc_hash': 'q', 'target_hash': 'a'}
    result = {'samples': {'math': [row, {**row, 'filter': 'strict-match', 'exact_match': 0}]}}
    assert sample_scores(result)[('math', 3)]['correct']
    result['samples']['math'].append(row)
    with pytest.raises(ValueError, match='Duplicate'):
        sample_scores(result)


def test_numeric_grading_ignores_formatting_without_scanning_prose():
    assert numeric_answer('$7.00.') == numeric_answer('7')
    assert numeric_answer('-1,234.50') == numeric_answer('-1234.5')
    assert numeric_answer('7.5') != numeric_answer('75')
    for invalid in ('', '[invalid]', '7 or 8', '12,34', 'The answer is 7', 'NaN'):
        assert numeric_answer(invalid) is None
    row = {'doc_id': 0, 'filter': 'flexible-extract', 'exact_match': 0,
           'doc_hash': 'q', 'target_hash': 'a', 'target': 'Calculation\n#### 7',
           'filtered_resps': ['$7.00.']}
    scored = sample_scores({'samples': {'gsm8k_cot_zeroshot': [row]}})
    assert scored[('gsm8k_cot_zeroshot', 0)]['correct']
    assert not scored[('gsm8k_cot_zeroshot', 0)]['official_correct']
