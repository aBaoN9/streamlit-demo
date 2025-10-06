import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import parse_year, parse_duration, primary_genre

from src.features import parse_year, parse_duration, primary_genre

def test_parse_year():
    assert parse_year("(2015–2022)") == 2015

def test_parse_duration():
    assert parse_duration("1 h 40 min") == 100
    assert parse_duration("356 min") == 356

def test_primary_genre():
    assert primary_genre("Action, Thriller") == "Action"
