import nox


@nox.session(venv_backend="none")
def format(session: nox.Session):
    """Format code using isort and black."""
    session.run("isort", ".")
    session.run("black", ".")
