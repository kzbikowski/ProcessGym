from scipy import stats
import json
import random
import copy
import numpy as np
import pandas as pd


class ProcessDataLoader:
    def __init__(self, simulation_config_filename, resource_eligibility_filename):
        with open(simulation_config_filename, "r") as read_file:
            self.simulation_config_data = json.load(read_file)
        with open(resource_eligibility_filename, "r") as read_file:
            self.resource_eligibility = json.load(read_file)

    def load_config_file(self, filename):
        with open(filename, "r") as read_file:
            self.simulation_config_data = json.load(read_file)

    def load_resource_eligibility(self, filename):
        with open(filename, "r") as read_file:
            self.simulation_config_data = json.load(read_file)

    def load_max_time(self):
        return self.simulation_config_data["max_simulation_time"]

    def load_available_resources(self):
        return self.simulation_config_data["available_resources"]

    def load_process_case_probability(self):
        return self.simulation_config_data["process_case_probability"]

    def load_queue_capacity_modifier(self):
        return self.simulation_config_data["queue_capacity_modifier"]

    def load_process_data(self):
        processes = []
        for process in self.simulation_config_data["loaded_processes"]:
            with open(process["filename"], "r") as read_file:
                data = json.load(read_file)
                processes.append(self.transform_data(data, process["frequency"], process["reward"]))
        return processes

    def transform_data(self, data, frequency, reward):
        process_id = data["process_id"]
        tasks = []
        for task in data["tasks"]:
            transitions = [Transition(*x.values()) for x in task["transitions"]]
            new_task = Task(task["id"],
                            process_id,
                            transitions,
                            task["duration"],
                            task["duration_SD"],
                            self._find_eligible_resources(task, process_id),
                            task["start"])
            tasks.append(new_task)
        return Process(process_id, tasks, frequency, reward)

    def _find_eligible_resources(self, task, process_id):
        for x in self.resource_eligibility:
            if x["process_id"] == process_id:
                for y in x["resource_eligibility"]:
                    if y["task_id"] == task["id"]:
                        return y["eligible_resources"]


class Transition:
    def __init__(self, task_id, probability):
        self.task_id = task_id
        self.probability = probability

    def __repr__(self):
        return f"Transition(task_id={self.task_id},probability={self.probability})"


class Process:
    def __init__(self, id, tasks, frequency=0, reward=0):
        self.id = id
        self.frequency = frequency
        self.reward = reward
        self.tasks = tasks

    def pick_next_task(self, task):
        if not task.transitions:
            return
        next_transition = random.choices(task.transitions, weights=tuple(x.probability for x in task.transitions), k=1)[0]
        next_task = self.get_task_by_id(next_transition.task_id)
        next_task.initialize_duration()
        next_task.start = False
        next_task.start_step = task.start_step
        return next_task

    def get_task_by_id(self, task_id):
        for task in self.tasks:
            if task.id == task_id:
                return copy.copy(task)

    def initialize_process_case(self, step):
        for task in self.tasks:
            if task.start:
                task.start_step = step
                task.initialize_duration()
                return copy.copy(task)

    def __repr__(self):
        return f"Process(id={self.id}, tasks={self.tasks}, frequency={self.frequency}, reward={self.reward})"


class Task:
    def __init__(self, id, process_id, transitions, duration_mean, duration_sd, eligible_resources, start=False):
        self.id = id
        self.transitions = transitions
        self.duration_mean = duration_mean
        self.duration_sd = duration_sd
        self.eligible_resources = eligible_resources
        self.start = start
        self.process_id = process_id
        self.duration = None
        self.allocated_resource = None
        self.start_step = None

    def initialize_duration(self):
        self.duration = round(stats.truncnorm.rvs(-1, 1, loc=self.duration_mean, scale=self.duration_sd, size=1)[0], 0)

    def __repr__(self):
        return f"Task(id={self.id}, process_id={self.process_id} transitions={self.transitions}, duration_mean={self.duration_mean}, duration_sd={self.duration_sd}, eligible_resources={self.eligible_resources}, start={self.start}) "


class Simulation:
    def __init__(self, available_resources, processes, process_case_probability, num_of_initial_process_cases, queue_capacity_modifier, state_repr):
        self.processes = processes
        self.processes_frequency = {x.id: x.frequency for x in processes}
        self.processes_rewards = {x.id: x.reward for x in processes}
        self.process_case_probability = process_case_probability
        self.enabled_tasks = []
        self.current_tasks = []
        self.resource_eligibility = []
        self.state_repr = state_repr
        self.all_resources = list(available_resources)
        self.available_resources = list(available_resources)
        self.completed_counter = self.step_counter = 0
        self.task_count = sum(len(process.tasks) for process in self.processes)
        self.action_space = [len(self.all_resources), self.task_count]
        self.all_tasks = [task for process in self.processes for task in process.tasks]

        self.queue_capacity = queue_capacity_modifier * 60 * len(self.all_resources)

        for resource in sorted(self.all_resources):
            temp = list()
            for task in sorted(self.all_tasks, key=id):
                temp.append(1) if str(resource) in task.eligible_resources.keys() else temp.append(-1)
            self.resource_eligibility.extend(temp)

        for _ in range(num_of_initial_process_cases):
            process_choice = random.choice(self.processes)
            self.enabled_tasks.append(process_choice.initialize_process_case(self.step_counter))

    def step(self, action):
        reward = 0
        [resource, task_id] = action
        is_enabled = False

        if random.random() < self.process_case_probability and not self._is_queue_full:
            self._pick_and_initialize_new_process_case()

        chosen_task = next(filter(lambda t: t.id == task_id, self.enabled_tasks), None)
        if not action == [-1, -1]:
            is_enabled = (self._number_of_enabled_tasks(task_id) > 0)
        if is_enabled and (resource in self.available_resources) and (str(resource) in chosen_task.eligible_resources.keys()):
            reward = self._assign_resource(chosen_task, resource)

        self._handle_completed_tasks()
        return self.state, reward

    def step_fifo(self):
        return self._step_heuristic("fifo")

    def step_spt(self):
        return self._step_heuristic("spt")

    def reset(self):
        self.enabled_tasks.clear()
        self.current_tasks.clear()
        self.available_resources = list(self.all_resources)
        self.completed_counter = 0
        self.step_counter = 0
        return self.state

    @property
    def state(self):
        if self.state_repr == "std":
            return self._state_std()
        elif self.state_repr == "a1":
            return self._state_a1()
        elif self.state_repr == "a2":
            return self._state_a2()
        elif self.state_repr == "a10":
            return self._state_a10()

    def _state_std(self):
        state = pd.DataFrame(-1, index=self.all_resources, columns=range(0, self.task_count))
        for task in self.current_tasks:
            state.loc[task.allocated_resource, :] = -1
            state.loc[task.allocated_resource, task.id] = 1
        for resource in self.available_resources:
            for process in self.processes:
                for task in process.tasks:
                    if str(resource) in task.eligible_resources.keys() and next(
                            (x for x in self.enabled_tasks if x.id == task.id), None):
                        state.loc[resource, task.id] = 0
                    else:
                        state.loc[resource, task.id] = -1
        return state

    def _state_a1(self):
        num_of_resources = len(self.all_resources)
        state_resource_list = [-1] * num_of_resources
        state_task_list = [0] * self.task_count
        for task in self.current_tasks:
            state_resource_list[task.allocated_resource] = task.id
        for i in range(self.task_count):
            state_task_list[i] = len([x for x in self.enabled_tasks if x.id == i])
        return pd.Series(state_resource_list + state_task_list)

    def _state_a2(self):
        num_of_resources = len(self.all_resources)
        state_resource_list = [-1] * num_of_resources
        state_task_list = [0] * self.task_count
        state_eligibility_list = []
        for task in self.current_tasks:
            state_resource_list[task.allocated_resource] = task.id
        for i in range(self.task_count):
            state_task_list[i] = len([x for x in self.enabled_tasks if x.id == i])
        return pd.Series(state_resource_list + state_task_list + self.resource_eligibility)

    def _state_a10(self):
        num_of_resources = len(self.all_resources)
        state_resource_list = [-1] * num_of_resources
        state_task_list = [0] * self.task_count
        for task in self.current_tasks:
            state_resource_list[task.allocated_resource] = task.id
        for i in range(self.task_count):
            state_task_list[i] = round(len([x for x in self.enabled_tasks if x.id == i]) / len(self.enabled_tasks),
                                       4) if self.enabled_tasks else 0
        return pd.Series(state_resource_list + state_task_list)

    def _step_heuristic(self, heuristic_name):
            sort_attr = ""
            if heuristic_name == "fifo":
                sort_attr = "start_step"
            elif heuristic_name == "spt":
                sort_attr = "duration"

            reward = 0
            action = []

            if random.random() < self.process_case_probability and not self._is_queue_full:
                self._pick_and_initialize_new_process_case()

            is_looping = True
            if self.enabled_tasks:
                for task in sorted(self.enabled_tasks, key=lambda x: getattr(x, sort_attr)):
                    for resource in self.available_resources:
                        if str(resource) in task.eligible_resources:
                            action = [resource, task.id]
                            reward = self._assign_resource(task, resource)
                            is_looping = False
                            break
                    if not is_looping:
                        break

            self._handle_completed_tasks()
            self.step_counter += 1
            if not action:
                action = [-1, -1]
            return self.state, reward, action

    def _pick_and_initialize_new_process_case(self):
        process_choice = random.choices(self.processes, [x.frequency for x in self.processes], k=1)[0]
        self.enabled_tasks.append(process_choice.initialize_process_case(self.step_counter))

    def _handle_completed_tasks(self):
        for task in self.current_tasks:
            task.duration -= 1
            if task.duration == 0:
                self.available_resources.append(task.allocated_resource)
                if not task.transitions:
                    self.completed_counter += 1
                else:
                    process = self._find_process_by_id(task.process_id)
                    self.enabled_tasks.append(process.pick_next_task(task))
                self.current_tasks.remove(task)

    def _find_process_by_id(self, id):
        for process in self.processes:
            if process.id == id:
                return process

    def _assign_resource(self, task, resource):
        self.available_resources.remove(resource)
        self.enabled_tasks.remove(task)
        task.allocated_resource = resource
        task.duration = round(task.duration * task.eligible_resources[str(resource)])
        self.current_tasks.append(task)
        return self.processes_rewards[task.process_id] if not task.transitions else 0

    def _number_of_enabled_tasks(self, task_id):
        return self.state.iloc[len(self.all_resources) + task_id]

    @property
    def _is_queue_full(self):
        return self.queue_capacity <= sum([x.duration for x in self.enabled_tasks])



