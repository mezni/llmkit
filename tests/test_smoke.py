import llmkit


def test_package_exposes_version() -> None:
    assert isinstance(llmkit.__version__, str)
