import numpy as np

from bananas.problem import choose_action, Agent, Environment


class MockEnvironment(Environment):
    state_size = 2
    action_size = 4

    def __init__(self):
        super().__init__(name='mock environment')

    def reset(self):
        pass

    def step(self, action):
        return 0


class MockAgent(Agent):
    description = 'Mock agent'

    def eval(self, state):
        return np.array([10, 0, -8.5, 4.9])

    @property
    def is_trained(self):
        return True

    @property
    def model_class(self):
        pass

    def _save(self, filename):
        pass


agent = MockAgent(MockEnvironment)


class TestChooseAction:
    def test_choose_best(self):
        assert choose_action(agent, None, 0) == 0

    def test_choose_random(self):
        times = 1000

        expected = times / agent.action_size
        counts = [0] * agent.action_size
        for _ in range(times):
            counts[choose_action(agent, None, 1)] += 1
        for count in counts:
            assert count < expected * 1.2
            assert count > expected * 0.8


def test_train():
    agent = MockAgent(MockEnvironment())
    agent.train()
    assert agent.is_trained
