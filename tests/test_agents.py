from functools import partial

from bananas import logger
from bananas.agents import DQN_Agent
from bananas.model import DuelingQNetwork
from bananas.problem import moving_average, moving_average_cross, loggable


def test_dqn_1(banana_environment):
    logger.info('Banana environment')

    agent = DQN_Agent(
        banana_environment,
        max_train_episodes=10)
    agent.train()

    assert True


def test_dqn_2(banana_environment):
    logger.info('Banana environment')

    agent = DQN_Agent(
        banana_environment,
        is_finished_function=[loggable(moving_average_cross, 50, 100), loggable(moving_average, 100, 13)],
        is_solved=loggable(moving_average, 100, 13),
        max_train_episodes=10)
    agent.train()

    assert True


def test_dueling_dqn_1(banana_environment):
    logger.info('Banana environment')

    agent = DQN_Agent(
        banana_environment,
        is_finished_function=[loggable(moving_average_cross, 50, 100), loggable(moving_average, 100, 13)],
        is_solved=loggable(moving_average, 100, 13),
        max_train_episodes=10,
        partial_network=partial(DuelingQNetwork, fc1_units=32, fc2_units=32,
                                value_fc1_units=32,
                                advantage_fc1_units=32)
    )
    agent.train()

    assert True


def test_load_agent(banana_environment):
    from bananas.problem import load
    agent2 = load(
        '/Users/claudio/Dropbox/udacity-drl/code/drlnd-p1-navigation/mlruns/0/70b2c1ddd5c441aa88f39955310725d1/artifacts/model_final.pkl',
        banana_environment)
    assert True