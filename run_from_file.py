from tensorflow import keras
from simprocess import ProcessDataLoader, Simulation
import dqn_learning as dqnl

nmb_of_test_episodes = 100
nmb_of_iterations_per_episode = 100

keras.backend.clear_session()

dir_name = "results_20210302_22_08/"

model = keras.models.load_model(dir_name + "last_model")
loader = ProcessDataLoader(dir_name+"conf/simulation_config.json", dir_name+"conf/resource_eligibility.json")

processes = loader.load_process_data()
available_resources = loader.load_available_resources()
process_case_probability = loader.load_process_case_probability()
queue_duration_limit = loader.load_queue_capacity_modifier()

env = Simulation(available_resources, processes, process_case_probability, 1, queue_duration_limit, "a10")

action_space = env.action_space

test_rewards = []

model.summary()

for e in range(nmb_of_test_episodes):
    state = env.reset().to_numpy()
    test_episode_reward_sum = 0
    train_episode_model_actions_count = {}

    for step in range(nmb_of_iterations_per_episode):
        state, reward, action, action_type = dqnl.epsilon_greedy_policy(model, env, state, action_space[1])
        if action_type == "q":
            if action in train_episode_model_actions_count:
                train_episode_model_actions_count[action] += 1
            else:
                train_episode_model_actions_count[action] = 1
        state = state.to_numpy()
        test_episode_reward_sum += reward
    test_rewards.append(test_episode_reward_sum)

    print("Episode test: {}, rewards: {}".format(e, test_episode_reward_sum), end="\n")
    print(train_episode_model_actions_count, end="\n")

    test_avg_rewards = sum(test_rewards) / len(test_rewards)

print(test_avg_rewards)
