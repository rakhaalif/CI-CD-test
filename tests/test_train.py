from src.train import train


def test_training_runs_and_is_reasonable():
    model, metrics = train(random_state=0)
    assert hasattr(model, "predict")
    assert metrics["accuracy"] >= 0.80
