import random
from simprocess import ProcessDataLoader, Simulation
import numpy as np
import time
import math

def get_action_from_int(int_action_value, nmb_of_tasks):
    """
    :param nmb_of_tasks: number of tasks
    :param nmb_of_resources: number of resources
    :param int_action_value: action value from the range NxM where N - nmb_of_resources, M - nmb_of_tasks
    :return: [resource, task] Vector of length 2 of actions to be taken by the environment
    """
    if int_action_value == 0:
        return [-1, -1]
    else:
        action_coded_value = int_action_value - 1
        resource = math.floor(action_coded_value / nmb_of_tasks)
        task = action_coded_value % nmb_of_tasks
        return [resource, task]


def main():
    loader = ProcessDataLoader("./conf/simulation_config.json", "./conf/resource_eligibility.json")
    processes = loader.load_process_data()
    available_resources = loader.load_available_resources()
    process_case_probability = loader.load_process_case_probability()
    queue_duration_limit = loader.load_queue_capacity_modifier()
    simulation = Simulation(available_resources, processes, process_case_probability, 1, queue_duration_limit, "a2")
    action_space = simulation.action_space
    nmb_of_episodes = 100
    nmb_of_steps_per_episode = 1500


    rewards = []
    for episode in range(nmb_of_episodes):
        reward_sum = 0
        start_time = time.time()
        state = simulation.reset()
        for step in range(nmb_of_steps_per_episode):
            # action = np.random.randint(action_space[0] * action_space[1] + 1)
            # action = get_action_from_int(action, action_space[1])
            # next_state, reward = simulation.step(action)
            next_state, reward, action = simulation.step_fifo()
            reward_sum += reward
        print("Episode: {}, rewards: {}, time: {}".format(episode, reward_sum, time.time() - start_time), end="\n")
        rewards.append(reward_sum)

    print("reward: {}".format(sum(rewards)/len(rewards)))


if __name__ == '__main__':
    main()
