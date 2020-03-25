import pytest
from unityagents import UnityEnvironment

from bananas.environment import UnityAdapter



@pytest.fixture(scope="session")
def banana_environment():
    return UnityAdapter(
            UnityEnvironment(
                file_name="/Users/claudio/Dropbox/udacity-drl/code/deep-reinforcement-learning/p1_navigation/Banana.app",
                no_graphics=True
            ),
            name='bananas')