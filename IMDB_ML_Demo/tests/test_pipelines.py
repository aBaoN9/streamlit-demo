from pathlib import Path

def test_models_exist():
    assert Path("models/decision_tree_rating_regressor.pkl").exists()
    assert Path("models/naive_bayes_genre_from_description.pkl").exists()
