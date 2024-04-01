# Name: robot-imitation
# Version: 2.0
# Authors: - Dimitrios papageorgiou
#          - Alexandros Georgantzas
#          - Kostis Kakkavas
#
# Implementation for imitation learning in the recognition of a certain state space for autonomous drones


# -= Importing needed Libraries =- #

# System libraries
from PPO_agent import PPOAgent, Transition
from Drone import Drone
from PID import PID

# Python Libraries
import numpy as np
import random
import math

# -= Importing needed Libraries =- #

# Initialization Part

env = Drone() # Initializing the Drone

# Initializing the reinforcement learning agent
agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n)

agent.load_weights('loop-weights') # Load the pre-trained neural network model

######################
#   Training Loop    #
# Uncomment to train #
######################

'''

# Tune settings for main training loop
solved = False
episode_count = 0
episode_limit = 5000

# -= Main Training Loop =- #

while not solved and episode_count < episode_limit: # Loop for episode count

    # Reset robot and get starting observations
    observation = env.reset()

    # Initialize episode data
    env.episode_score = 0
    time_counter = 0
    model_action = 2
    flag = True

    # Imitation PID Initialization and tuning
    zPID = PID()
    zPID.init(13000 - random.random() * 5000, 0, 1500, 200) # Altitude PID
    xPID = PID()
    xPID.init(0.1, 0, 0.125, 200) # Position PID
    thiPID = PID()
    thiPID.init(35, 0, 42, 200) # Angle PID

    # Set targets for altitude and position
    ztarget = random.random() + 0.5
    xtarget = 2 * 2 * (random.random() - 0.5)
    noise = 0 * np.random.normal(0, 0.1) # Noise for pre-training (currently 0)

    # Log the targets, non-noised and noised
    print("Goal z: ", ztarget)
    print("Goal x: ", xtarget)
    print("Noised z: ", ztarget + noise)
    print("Noised x: ", xtarget + noise)

    for step in range(env.steps_per_episode): # Loop for every time step

        # Proceed to run the imitation sub-routine for the first episodes to train the model
        if episode_count <= 1000:

            # Get data for time step for a given action
            new_observation, reward, done, info = env.step([model_action], ztarget, xtarget)

            # Recall current position
            drone_z = env.robot.getPosition()[2]
            drone_x = env.robot.getPosition()[0]
            drone_phi = new_observation[1]

            if flag: # Run altitude PID with saturation filter
                value = zPID.PID_calc(ztarget + noise, drone_z)

                if value > 558.586 + 20:
                    value = 558.586 + 20
                elif value < 558.586 - 20:
                    value = 558.586 - 20

                model_action = int(math.ceil(value - 558.586 + 20) * 4 / 40)

                flag = False
            else: # Run position PID with saturation filter

                thi = xPID.PID_calc(xtarget + noise, drone_x)
                dom = thiPID.PID_calc(thi, drone_phi)

                if dom < 0:
                    model_action = 5
                else:
                    model_action = 6

                flag = True

            # Store the imitated observations, actions, action probabilities, reward and new observations
            trans = Transition(observation, model_action, 0.7 + 0.2 * random.random(), reward, new_observation)
            agent.store_transition(trans)
        else: # Proceed to run the normal learning method (PPO Agent)
            selected_action, action_prob = agent.work(observation, type_="selectAction")
            new_observation, reward, done, info = env.step([selected_action], ztarget, xtarget)

            # Store the imitated observations, actions, action probabilities, reward and new observations
            trans = Transition(observation, selected_action, action_prob, reward, new_observation)
            agent.store_transition(trans)

        if done: # Train the model when the episode is done
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step+1)
            solved = env.solved(episode_count)  # Check whether the task is solved
            break

        time_counter += 1 # Increment time step counter
        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation

    # Log episode scores and other data
    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter
    print("Time counter: ", time_counter)

    # if episode_count == 1000:
    #     agent.save_weights('7000') # Save the already trained weights of the neural network

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

'''

###############################
#   Trained Execution Loop    #
###############################

# Tune settings for trained execution
observation = env.reset()
env.episode_score = 0.0
time = 0

while True: # Main loop
    selected_action, action_prob = agent.work(observation, type_="selectActionMax") # Select actions based on the best training results

    h0 = 0.7  # Initial lift off height
    r = 1     # Circle radius
    T = 20    # Circle period

    # Lift up to altitude z = h0
    if time > 2:
        observation, _, done, _ = env.step([selected_action], h0, 0)

    # Perform the circle motion
    observation, _, done, _ = env.step([selected_action], 2 * h0 - r * math.cos(4 * math.pi * time / T), r * math.sin(4 * math.pi * time / T))

    time += 0.032

    if done: # Run the main loop until one of the space termination criteria is met
        observation = env.reset()