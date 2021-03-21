<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#configuration-files-json-schemas">Configuration files JSON schemas</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
For the purpose of using tabular and approximate algorithms in the area of reinforcement learning, we designed and developed a dedicated simulation environment that we call ProcessGym.
It can serve as a general-purpose framework for testing resource allocation algorithms.



<!-- GETTING STARTED -->
## Getting Started

Ensure you use Python 3.8. \
To be able to run this project you just need to clone this repo and install all the requirements from the **`requirements.txt`** file.

[comment]: <> (### Prerequisites)


<!-- USAGE EXAMPLES -->
## Usage

`python test.py`

`python dqn_learning.py`

**`test.py`** file is used for basic testing of prepared simulation configurations and defined business processes.  
One of 3 resource allocation methods can be used in the test file: random, FIFO (first in first out), and SPT (shortest processing time). 
`Simulation` object has 3 methods for this purpose. Respectively `step()`, `step_fifo()` and `step_spt()`  .
The `step()` method requires an action - a two-element list to be passed to it. List has to consist of the id of the task to which we will allocate the resource and allocated resource.  
`test.py` file, also allows to specify duration of the entire simulation by modyfing `nmb_of_episodes` and `nmb_of_steps_per_episode` variables. 

Running **`dqn_learning.py`** trains and tests the double DQN model. In **`dqn_learning.py`** user can control duration of training and testing.
Variables responsible for that are:   `nmb_of_train_episodes`, `nmb_of_test_episodes`, `nmb_of_iterations_per_episode`, `nmb_of_episodes_before_training`

Two configuration files must be passed to the constructor of the `ProcessDataLoader` class. One defines the simulation environment itself, 
and the other defines resource eligibilites (schemas of the files are specified in section below).

 

## Configuration files JSON schemas
Examples of config files are in **conf** directory.

### Simulation config
```json
{
  "title": "Simulation config",
  "type": "object",
  "properties": {
    "process_case_probability": {
      "description": "Probability of new process case arriving in each step",
      "type": "number"
    },
    "queue_capacity_modifier": {
      "description": "Modifier limiting size of enabled_tasks queue",
      "type": "number"
    },
    "available_resources": {
      "description": "List of available resources",
      "type": "array",
      "items": {
        "type": "number"
      },
      "loaded_processes": {
        "description": "List of processes definitions to be loaded",
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "filename": {
              "description": "Path to process definition file",
              "type": "string"
            },
            "frequency": {
              "description": "Relative frequency of process case appearance",
              "type": "number"
            },
            "reward": {
              "description": "Reward for completing process case",
              "type": "number"
            }
          }
        }
      }
    }
  }
}
```
### Process definition schema
```json
{
  "title": "Process definition",
  "type": "object",
  "properties": {
    "process_id": {
      "description": "Unique identifier of business process",
      "type": "number"
    },
    "tasks": {
      "description": "List of tasks",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "description": "Unique task identifier",
            "type": "number"
          },
          "duration": {
            "description": "Average task duration",
            "type": "number"
          },
          "duration_sd": {
            "description": "Standard deviation of task duration",
            "type": "number"
          },
          "start": {
            "description": "Flag indicating whether business process starts with this task",
            "type": "boolean"
          },
          "transitions": {
            "description": "List of possible task transitions",
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": {
                  "description": "Task identfier",
                  "type": "number"
                },
                "probability": {
                  "description": "Probability of transitioning to task",
                  "type": "number"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### Resource eligibility config
```json
{
  "title": "Resource eligibilities",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "resource_eligibility": {
        "description": "List of eligible resources for tasks",
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "task_id": {
              "description": "Task identifier",
              "type": "number"
            },
            "eligible_resources": {
              "type": "object",
              "properties": {
                "_resource_id": {
                  "description": "Task duration modifier (_resource_id must be a number)",
                  "type": "number"
                }
              }
            }
          }
        }
      }
    }
  }
}
```




<!-- ROADMAP -->

[comment]: <> (## Roadmap)

[comment]: <> (See the [open issues]&#40;https://github.com/othneildrew/Best-README-Template/issues&#41; for a list of proposed features &#40;and known issues&#41;.)



<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request



[comment]: <> (<!-- LICENSE -->)

[comment]: <> (## License)

[comment]: <> (Distributed under the MIT License. See `LICENSE` for more information.)



<!-- CONTACT -->
## Contact



[comment]: <> (<!-- ACKNOWLEDGEMENTS -->)

[comment]: <> (## Acknowledgements)



