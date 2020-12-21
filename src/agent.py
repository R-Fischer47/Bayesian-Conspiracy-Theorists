import itertools
import math
import statistics
from typing import List, TYPE_CHECKING

import scipy.stats
from mesa import Agent

if TYPE_CHECKING:
    from model import ConspiracyModel

DISTRIBUTION_MEAN = .5


class ConspiracyAgent(Agent):
    # Extra type hint so we know get completion for members of ConspiracyModel
    model: 'ConspiracyModel'

    connections: List['ConspiracyAgent']

    # Bayesian stuff
    prior_value: float
    prior_history: List[float]
    prior_confidence: float
    prior_sd: float
    p_h: float

    def __init__(self, unique_id: int, model: 'ConspiracyModel') -> None:
        super().__init__(unique_id, model)

        self.prior_history = self._generate_initial_belief()
        self._recompute_belief()
        self.p_h = self.model.start_p_h

    def update_connections(self):
        neighbors = list(self.model.grid.iter_neighborhood(self.pos, moore=False, radius=self.model.agent_range))
        # print(neighbors)

        self.connections = [
            agent
            for agent
            in self.model.agents
            if (agent != self
                and agent.pos in neighbors
                and self.agrees_with(agent))
        ]

    # def can_reach(self, agent: 'ConspiracyAgent') -> bool:
    #     """
    #     Returns whether the agent can physically reach another agent, based on the distance between them.
    #     :param agent: The other agent.
    #     :return: Whether the agent can communicate.
    #     """

    def agrees_with(self, agent: 'ConspiracyAgent') -> bool:
        """
        Returns whether the agent is willing to communicate with another agent.
        :param agent: The other agent.
        :return: Whether the agents agree.
        """
        lower = self.prior_value - self.prior_sd
        upper = self.prior_value + self.prior_sd
        return lower <= agent.prior_value <= upper

    def communicate(self) -> None:
        """
        Communicate with all connected agents, updating own belief based on the beliefs of the connected agents.
        """
        connection_mean_prior = statistics.mean(agent.prior_value for agent in self.connections)
        self.prior_history.append(connection_mean_prior)

        self._bayesian_update(connection_mean_prior)
        self._recompute_belief()

    def _bayesian_update(self, new_evidence: float) -> None:
        """
        Update confidence in own belief, based on new evidence.
        :param new_evidence: Belief of connected agents.
        """

        true_distribution: scipy.stats.norm_gen = scipy.stats.norm(DISTRIBUTION_MEAN, self.model.initial_sd)
        p_e = true_distribution.pdf(new_evidence)

        agent_belief_distribution: scipy.stats.norm_gen = scipy.stats.norm(self.prior_value, self.prior_sd)
        p_e_given_h = agent_belief_distribution.pdf(new_evidence)

        p_h_given_e = (p_e_given_h * self.p_h) / p_e

        self.p_h = p_h_given_e

    def _recompute_belief(self) -> None:
        """
        Calculate prior_value, prior_sd and prior_confidence after a change to prior_history.
        """
        self.prior_value = statistics.mean(self.prior_history)
        self.prior_sd = statistics.stdev(self.prior_history)
        self.prior_confidence = (1 / self.prior_sd * math.sqrt(2 * math.pi))

    def _generate_initial_belief(self) -> List[float]:
        """
        Create a list of prior_sample_size length containing random samples from a gaussian distribution.
        Ensures that 0. <= value <= 1.
        :return: The list of random values.
        """

        infinite_random_generator = iter(lambda: self.random.gauss(DISTRIBUTION_MEAN, self.model.initial_sd), None)
        valid_values = filter(lambda random_value: 0. <= random_value <= 1., infinite_random_generator)

        # Take prior_sample_size values from the generator
        return list(itertools.islice(valid_values, self.model.prior_sample_size))

    def step(self) -> None:
        self.update_connections()

        if self.connections:
            self.communicate()
