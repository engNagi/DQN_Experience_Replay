from DQN import DQN
from Utils import Utils
from datetime import datetime

import gym
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import time

MAX_Exp = 500000
Min_Exp = 500
Target_Update_period = 10000
action_size = 4 # env.action_space.n

if __name__ == "__main__":
    # hyper parameters
    gamma = 0.99
    batch_sz = 32
    num_episodes = 100
    total_t = 0
    experience_replay_buffer = []
    episode_rewards = np.zeros(num_episodes)
    last_100_avgs = []
    learning_rate = 0.00025

    # epsilon for Epsilon Greedy Algorithm
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000

    # create Atari Environment
    env = gym.envs.make("Breakout-v0")


# Create original and Target Network
    model = DQN(learning_rate=learning_rate, network_name="model", actions_num=action_size)
    target_model = DQN(learning_rate=learning_rate, network_name="target_model", actions_num=action_size)

    with tf.Session() as sess:
        model.create_session(sess)
        target_model.create_session(sess)
        sess.run(tf.global_variables_initializer())
        model.chk_pnt_load()

        print("... Pushing into experience_replay_buffer....")
        # reseting the Atari Env.
        obs =env.reset()
        # preprocessing the frames
        state = Utils(obs)._stack_frames()

        # Main loop for training and Filling experience Replay
        for i in tqdm.tqdm(range(Min_Exp)):
            time.sleep(0.01)
            env.render()
            # Choosing random action from the available actions
            action = np.random.randint(0, action_size)
            # Performing the chosen action and receiving back the ENV_observation, reward,
            # final state "Done"->(if the game is over) and information "_"
            obs, reward, done, _ = env.step(action)

            # preprocessing the next observation
            next_state = Utils(obs)._stack_frames()

            #filling our experience replay
            experience_replay_buffer.append((state, action, reward, next_state, done))

            #if the game ended process the last frame else set the state to the next state
            if done:
                obs = env.reset()
                obs_small =Utils(obs)._stack_frames()

            else:
                state = next_state

        # Play a number of episodes to learn
        for i in tqdm.tqdm(range(num_episodes)):
            env.render()
            time.sleep(0.01)
            t0 = datetime.now()


            #reset the Evironment
            obs = env.reset()
            state = Utils(obs)._stack_frames()
            assert (state.shape == (4, 84, 84))
            cost = None

            total_training_time = 0
            episode_steps = 0
            episode_reward = 0
            # Reset the Graph

            done = False
            while not done:
                env.render()
                #update target network
                if total_t % Target_Update_period == 0:
                    target_model.polyek_target_n_update(model.params)
                    print("Copied model parameters to target network. total_t = %s, period = %s" % (total_t,
                                                                                                Target_Update_period))
                # take action
                action =model.eps_explore(epsilon, state)
                obs, reward, done, _ = env.step(action)
                next_state = Utils(obs)._stack_frames()

                episode_reward +=reward

                #remove oldest expeience from the replay buffer
                if len(experience_replay_buffer) == MAX_Exp:
                    experience_replay_buffer.pop(0)

                #save the last expeience
                experience_replay_buffer.append((state, action, reward, next_state, done))

                #train the model
                t0_2 = datetime.now()
                loss = model.learn(model, target_model, experience_replay_buffer, gamma, batch_sz)
                dt = datetime.now() - t0_2

                total_training_time += dt.total_seconds()
                episode_steps += 1

                state = next_state
                total_t += 1

                epsilon = max(epsilon-epsilon_change, epsilon_min)

            duration = datetime.now() - t0

            episode_rewards[i] = episode_reward
            time_per_step = total_training_time / episode_steps

            last_100_avg = episode_rewards[max(0, i-100):i +1].mean()
            last_100_avgs.append(last_100_avgs)
            print("\n Episode:", i, "\n Duration:", duration, "\n Num steps:", num_episodes, "\n Reward:", episode_reward,
                "\n Training time per step:", "%.3f" % time_per_step, "\n Avg Reward (last 100):", "%.3f" % last_100_avg,
                "\n Epsilon:", "%.3f" % epsilon)


            if i % 50 == 0:
                model.chk_pnt_save(i)
            sys.stdout.flush()

        env.close()

        #plots
    plt.plot(last_100_avgs)
    plt.xlabel("Episodes")
    plt.ylabel("Average Rewards")
    plt.show()








