"""scrappyfit-cli: the console entry point is a headless batch fit, not the GUI."""
import pytest

from scrappyfit import cli, config

needs_db = pytest.mark.skipif(config.database_path(required=False) is None,
                              reason='no GeoPIXE database available')


def test_entry_point_is_the_cli_not_the_gui():
    """pyproject's scrappyfit-cli used to launch the window."""
    import pathlib
    text = (pathlib.Path(__file__).parents[1] / 'pyproject.toml').read_text()
    assert 'scrappyfit-cli = "scrappyfit.cli:main"' in text


@needs_db
def test_element_tokens_pick_the_shell():
    from scrappyfit.session import Session
    db = Session().db
    assert cli._pairs('C,O,FeL,Fe:L,PbM, Si', db) == [
        (6, 1), (8, 1), (26, 2), (26, 2), (82, 3), (14, 1)]
    with pytest.raises(SystemExit):
        cli._pairs('Xx', db)


def test_quantify_without_efficiency_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(['a.txt', '--elements', 'O', '--out', str(tmp_path),
                  '--quantify'])
    assert '--efficiency' in capsys.readouterr().err
