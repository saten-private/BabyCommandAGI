# ⭐️Now supported o1-preview(OpenRouter) and Claude 3.5 Sonnet(8K output)⭐️

# Precautions

- There is a risk of inadvertently damaging the environment. Please execute primarily in a virtual environment such as Docker.
- The objective may not be achieved and it may continue to loop. The amount of API usage may increase in such cases, so please use responsibly.
- It is recommended to use Claude 3.5 Sonnet or higher as it has been mainly verified with Claude 3.5 Sonnet or higher. (BabyCommandAGI uses the 8K output of [here](https://x.com/alexalbert__/status/1812921642143900036) when using Claude 3.5 Sonnet)
(If you are using GPT-4 Turbo, it is recommended to use v3 of the older BabyCommandAGI.)

# Objective

BabyCommandAGI is designed to test what happens when you combine CLI and LLM, which are older computer interaction interfaces than GUI. Those who are not familiar with computers may not know, but CLI is an old computer interaction interface. Even now, many computer operations can be done through CLI (common Linux servers mainly use CLI). Imagine LLM and CLI having a conversation. It's exciting to think about what could happen. I hope you will all try it out and find new use cases.

This system is recommended to be run with an API of Claude 3.5 Sonnet or higher.

This Python script system is based on [BabyAGI](https://github.com/yoheinakajima/babyagi). However, the part that was the thinking part of [BabyAGI](https://github.com/yoheinakajima/babyagi) has been greatly simplified in order to execute commands efficiently. (This may change later)

# Use Cases

BabyCommandAGI has the potential to be used in various cases. Please try using it to find use cases.

Below are some known useful use cases.

## Automatic Programming

Create an app automatically just by providing feedback

### Programming Examples

- Reversi
https://x.com/saten_work/status/1791550524988490053
- Snake Game
https://x.com/saten_work/status/1723509089471492563

## Automatic Environment Setup

- Install Flutter in a Linux container environment, create a Flutter app, launch a web server, and make it accessible from outside the container

https://twitter.com/saten_work/status/1667126272072491009

## Other

- Get weather forecast
https://x.com/saten_work/status/1791558481432232355

# Mechanism

This script works by executing the following continuous loop:

1. pull the next task from the task list. (It starts with one plan task.)
2. determine whether the task is a command task or a planned task
3. if it is a command task: 
    1. Execute the command. 
    2. If the Status Code of the command execution result is 0 (success):
        Go to 5.
    3. Otherwise (failure): 
        Analyze the history of executions with LLM and create a new task list according to the OBJECTIVE.
4. for plan tasks:
    1. plan with LLM based on the plan task, the history of executions and the OBJECTIVE, and create a new task list.
5. If user feedback is generated:
    1. plan and create a new task list in LLM based on the OBJECTIVE and  the history of executions while being aware of feedback.

![Architecture](docs/Architecture-20240718-2.png)

# Setup

Please follow the steps below:

1. ```git clone https://github.com/saten-private/BabyCommandAGI.git```
2. Enter the BabyCommandAGI directory with ```cd```.
3. Create a file to insert environment variables with ```cp .env.example .env```.
4. Set ANTHROPIC_API_KEY. (If you use OpenAI models, set OPENAI_API_KEY)
5. (Optional) Set the objective of the task management system to the OBJECTIVE variable.

# Execution (Docker)

As a prerequisite, docker and docker-compose must be installed. Docker desktop is the simplest option https://www.docker.com/products/docker-desktop/

## Run

```
docker-compose up -d && docker attach babyagi
```

## Stop

```
docker-compose stop
```

**Note: Even if you exit with Ctrl+C, it will not stop unless you run ```docker-compose stop``` or ```./clean.sh```. Please be careful.**

**Note: The agent might loop indefinitely if it cannot achieve its objective. Please be aware of the cost of Anthropic and OpenAI APIs usage.**

The AI's generated items will be created in the ```workspace``` folder.

If you fail, you can resume from where you left off by running it again.

Changing the OBJECTIVE will clear the list of future tasks and OBJECTIVE feedback.

## Feedback to AI

By entering "f", you can give the AI user feedback on the OBJECTIVE. This allows AI to feed back information that is not available from the CLI, such as the GUI.

## Answer while AI is executing a command

Normally, the AI cannot answer with a such as "y" or "n" to a command it is executing, but it will enter a mode where it can answer by entering "a".

(By the way, if a shell command waits for a answer like “y” or “n” for more than 5 minutes and the LLM thinks it is appropriate to answer, the LLM will automatically answer like “y” or “n” based on its judgment of the situation at that time.)

# Useful commands

- ```./clean.sh```

```workspace```, resets the environment (container). Also ```./new_store.sh``` also executes
- ```./backup_workspace.sh```

Backup your ``workspace`` by creating a folder with the current time in ``workspace_backup``.
(Note that environment (container) and BabyCommandAGI data will not be backed up.)
- ```./new_store.sh```

New BabyCommandAGI data (remembered information) will be created. Because of the switch to new data, BabyCommandAGI will not remember anything.

# Logs

The logs during execution are saved under the ```log``` folder.
The log file name is determined by the OBJECTIVE identifier and the RESULTS_STORE_NAME.

# Saved Data

The following are saved up to the point where they were executed:
- Tasks executed up to a certain point are saved under the ```data``` folder.
- The last current directory is under the ```pwd``` folder.
- The dump of the last environment variables is under the ```env_dump``` folder.

# Contributing

BabyCommandAGI is still in the early stages, determining its direction and the steps to get there. Currently, BabyCommandAGI is aiming for simplicity. To maintain this simplicity, when submitting PRs, we kindly ask you to follow the guidelines below:

- Focus on small, modularized fixes rather than large-scale refactoring.
- When introducing new features, provide a detailed explanation of the corresponding specific use cases.

Note from @saten-private (May 21, 2023):

I am not used to contributing to open source. I work another job during the day and I don't know if I can check PRs and issues frequently. However, I cherish this idea and hope it will be useful for everyone. Please feel free to let me know if there's anything. I am looking forward to learning a lot from you all.
I am a novice, I cannot speak English, and I barely know cultures outside of Japan. However, I cherish my ideas, and I hope they will be of use to many people.
(I'm sure I will continue to come up with many boring ideas in the future)

<h1 align="center">
  ✨ BabyCommandAGI's GitHub Sponsors ✨
</h1>
<p align="center">
  The maintenance of this project is made possible thanks to all of the following sponsors. If you'd like to become a sponsor and have your avatar logo displayed below, please <a href="https://github.com/sponsors/saten-private">click here</a>. 💖 You can become a sponsor for $5.
</p>
<p align="center">
<!-- sponsors --><!-- sponsors -->
</p>