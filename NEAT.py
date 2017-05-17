# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg
# NOTE: This was run using revision 1186029827c156e0ff6f9b36d6847eb2aa56757a of CodeReclaimers/neat-python, not a release on PyPI.

from __future__ import print_function

import multiprocessing
import os
import pickle
import random
import time
import argparse
import logging
import sys
import gym.wrappers as wrappers
import matplotlib.pyplot as plt
import neat
import numpy as np
import gym

import visualize


discounted_reward = 0.9
min_reward = -200
max_reward = 200
score_range = []


class NeatAC(object):
    def __init__(self, config):
        pop = neat.Population(config)
        self.stats = neat.StatisticsReporter()
        pop.add_reporter(self.stats)
        pop.add_reporter(neat.StdOutReporter(True))
        # Checkpoint every 10 generations or 900 seconds.
        pop.add_reporter(neat.Checkpointer(10, 900))
        self.config = config
        self.population = pop
        self.pool = multiprocessing.Pool()

    def execute_algorithm(self, generations):
        self.population.run(self.fitness_function, generations)

    def softmax(self, output):
        """
        Calculate the softmax of each output in output array
        :param output: 
        :return: 
        """
        return np.exp(output) / np.sum(np.exp(output), axis=0)

    def getActionUsingSoftmax(self, output):
        softmax = np.exp(output) / np.sum(np.exp(output), axis=0)

        p = random.uniform(0, 1)
        cumulative_probability = 0.0
        for n, prob in enumerate(softmax):
            cumulative_probability += prob
            if p <= cumulative_probability:
                return n

        # if did not return an index then pick random index
        return random.randint(0, len(output))



    def fitness_function(self, genomes, config):
        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))
            g.fitness = []

        episodes = []
        for genome, net in nets:
            observation = env.reset()
            episode_data = []
            total_reward = 0.0
            episode_count = 0
            while True:
                if net is not None:
                    # take action based on observation
                    nn_output = net.activate(observation)
                    action = self.getActionUsingSoftmax(nn_output)
                else:
                    # otherwise take a sample action
                    action = env.action_space.sample()

                # perform next step
                observation, reward, done, info = env.step(action)
                total_reward += reward
                episode_data.append((episode_count, observation, action, reward))

                if done:
                    break

                episode_count += 1

            episodes.append((total_reward, episode_data))
            # fitness is total score/episode_count
            genome.fitness = total_reward/episode_count


        scores = [score for score, episode in episodes]
        score_range.append((min(scores), np.mean(scores), max(scores)))

        print(min(map(np.min, score_range)), max(map(np.max, score_range)))






if __name__ == '__main__':
    # these belong to NEAT class
    # discounted_reward = 0.9
    # min_reward = -200
    # max_reward = 200
    # score_range = []
    # above belong to NEAT class


    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)
    print("action space: {0!r}".format(env.action_space))
    print("observation space: {0!r}".format(env.observation_space))

    # Limit episode time steps to cut down on training time.
    # 400 steps is more than enough time to land with a winning score.
    print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
    env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = 400
    print(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)

    # run the algorithm

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    agent = NeatAC(config)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    while 1:
        try:
            agent.execute_algorithm(4)

            visualize.plot_stats(agent.stats, ylog=False, view=False, filename="fitness.svg")

            if score_range:
                S = np.array(score_range).T
                plt.plot(S[0], 'r-')
                plt.plot(S[1], 'b-')
                plt.plot(S[2], 'g-')
                plt.grid()
                plt.savefig("score-ranges.svg")
                plt.close()

            mfs = sum(agent.stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))

            mfs = sum(agent.stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0}".format(mfs))

            # Use the five best genomes seen so far as an ensemble-ish control system.
            best_genomes = agent.stats.best_unique_genomes(5)
            best_networks = []
            for g in best_genomes:
                best_networks.append(neat.nn.FeedForwardNetwork.create(g, config))

            solved = True
            best_scores = []
            for k in range(100):
                observation = env.reset()
                score = 0
                while 1:
                    # Use the total reward estimates from all five networks to
                    # determine the best action given the current state.
                    total_rewards = np.zeros((3,))
                    for n in best_networks:
                        output = n.activate(observation)
                        total_rewards += output

                    best_action = np.argmax(total_rewards)
                    observation, reward, done, info = env.step(best_action)
                    score += reward
                    env.render()
                    if done:
                        break

                best_scores.append(score)
                avg_score = sum(best_scores) / len(best_scores)
                print(k, score, avg_score)
                if avg_score < 200:
                    solved = False
                    break

            if solved:
                print("Solved.")

                # Save the winners.
                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name + '.pickle', 'wb') as f:
                        pickle.dump(g, f)

                    visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
                    visualize.draw_net(config, g, view=False, filename="-net-enabled.gv",
                                       show_disabled=False)
                    visualize.draw_net(config, g, view=False, filename="-net-enabled-pruned.gv",
                                       show_disabled=False, prune_unused=True)

                break
        except KeyboardInterrupt:
            print("User break.")
            break

    env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
