"""Main test file for the streamlit app

Test can be run locally in the terminal with "pytest -v"

See <https://github.com/streamlit/llm-examples> for an exmaple of how to run the tests
with GitHub Actions.

"""

# Imports
import os

import pyprojroot
from streamlit.testing.v1 import AppTest

streamlit_pages_dir = pyprojroot.here() / "streamlit/pages"


def test_home():
    """ "Test that Home.py runs without error"""
    # Run the app
    at = AppTest.from_file("Home.py", default_timeout=30).run()

    # Check that it runs without error
    assert not at.exception

    # Check that "data_df" is in st.session_state
    assert "data_df" in at.session_state

    return


def test_all_pages():
    """Loop over all .py files in streamlit/pages/ and check for runtime errors."""
    for name in os.listdir(streamlit_pages_dir):
        path = os.path.join(streamlit_pages_dir, name)

        # Skip directories (including __pycache__) and non-.py files
        if not os.path.isfile(path) or not name.endswith(".py"):
            continue
        if name == "__init__.py":
            continue

        # Run the app
        at = AppTest.from_file(path, default_timeout=30).run()
        assert not at.exception, f"Exception raised in {name}"
