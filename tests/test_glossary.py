from dashboard import glossary


def test_glossary_non_empty():
    assert len(glossary.GLOSSARY) > 10


def test_glossary_keys_unique():
    assert len(glossary.GLOSSARY) == len(set(glossary.GLOSSARY))


def test_glossary_values_non_empty():
    assert all(v and v.strip() for v in glossary.GLOSSARY.values())


def test_glossary_covers_statuses_and_severity():
    for term in ["detected", "active", "ended", "recovering", "complete"]:
        assert term in glossary.GLOSSARY
    for term in ["unburned", "low", "moderate", "high", "very_high"]:
        assert term in glossary.GLOSSARY


def test_glossary_has_indices_and_metrics():
    for term in ["NDVI", "NDWI", "NBR", "dNBR", "FRP", "weather_index"]:
        assert term in glossary.GLOSSARY


def test_readme_intro_present():
    assert glossary.README_INTRO.strip()
