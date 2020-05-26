import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_adversaries = 2
        num_good_agents = 2
        num_landmarks = 2
        self.num_adversaries = num_adversaries
        self.num_good_agents = num_good_agents
        num_agents = num_adversaries + num_good_agents # deactivate "good" agent
        # add agents
        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False # last agent is good agent
            agent.size = 0.1 if agent.adversary else 0.15
            agent.accel = 3.0 if agent.adversary else 5
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 0.5 if agent.adversary else 2.0
            agent.action_callback = None if i < num_adversaries else self.prey_policy

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.15
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def prey_policy(self, agent, world):
        chosen_action = np.array([0,0], dtype=np.float32)
        return chosen_action

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0, 0, 0])
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark
    #     rew = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.adversary == False]
    #         rew -= min(dists)/self.num_good_agents
    #         if min(dists) < agent.size + world.landmarks[0].size:
    #             rew += 5/self.num_good_agents
    #     return rew

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark
    #     rew = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.adversary == False]
    #         rew -= min(dists)/self.num_good_agents
    #         if min(dists) < agent.size + world.landmarks[0].size:
    #             rew += 10/self.num_good_agents
    #     return rew

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.adversary == False]
            if min(dists) < world.landmarks[0].size + agent.size:
                rew += 10/self.num_good_agents
        return rew
    
    def reset_radius(self,sample_radius):
        sample_radius = max(sample_radius,1.5)
        self.sample_radius = sample_radius

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        landmark_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                landmark_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        adv_pos = []
        good_pos = []
        for other in world.agents:
            if other is agent: continue
            if other.adversary:
                adv_pos.append(other.state.p_pos - agent.state.p_pos)
            else:
                good_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + adv_pos + good_pos + landmark_pos)
