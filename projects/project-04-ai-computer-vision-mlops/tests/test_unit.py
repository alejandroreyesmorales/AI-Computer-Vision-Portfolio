
from src.classification_utils import classify_probability


def test_probability_below_threshold_is_normal():
    prediction, class_id = classify_probability(0.2)

    assert prediction == "Normal"
    assert class_id == 0


def test_probability_above_threshold_is_abnormal():
    prediction, class_id = classify_probability(0.8)

    assert prediction == "Abnormal"
    assert class_id == 1


def test_probability_at_threshold_is_abnormal():
    prediction, class_id = classify_probability(0.5)

    assert prediction == "Abnormal"
    assert class_id == 1
