
def classify_probability(probability):
    class_id = int(probability >= 0.5)
    prediction = "Abnormal" if class_id == 1 else "Normal"

    return prediction, class_id
