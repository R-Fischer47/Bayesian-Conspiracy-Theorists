import statistics
from typing import Any, List, Union

from matplotlib import pyplot
from mesa import Model, Agent
from mesa.space import SingleGrid
from mesa.time import RandomActivation

from agent import ConspiracyAgent


class ConspiracyModel(Model):

    def __init__(self,
                 n_agents: int,
                 width: int,
                 height: int,
                 agent_reach_radius: int,
                 prior_sample_size: int,
                 initial_sd: float,
                 start_p_h: float,
                 *args: Any,
                 **kwargs: Any) -> None:
        """
        Create the model.
        :param n_agents: Number of agents to place.
        :param width: Width of the grid.
        :param height: Height of the grid.
        :param agent_reach_radius: Radius around the agent in which it can connect.
        :param prior_sample_size: Size of initial belief sample.
        :param initial_sd: Initial standard deviation of the agents' beliefs.
        :param start_p_h: Initial p|h value.
        """
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents

        self.agent_range = agent_reach_radius

        self.prior_sample_size = prior_sample_size
        self.initial_sd = initial_sd
        self.start_p_h = start_p_h

        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

        print('Placing agents.')
        for i in range(self.n_agents):
            agent = ConspiracyAgent(i, self)
            self.schedule.add(agent)

            self.grid.position_agent(agent)
        print('Finished placing agents.')

    def step(self) -> None:
        self.schedule.step()

        print('Average confidence', statistics.mean(agent.prior_confidence for agent in self.agents))

        # Create a histogram: # TODO do this in the mesa webpage
        if self.schedule.time % 10 == 0:
            beliefs = [agent.prior_value for agent in self.agents]
            pyplot.hist(beliefs, bins=30)
            pyplot.show()

    @property
    def agents(self) -> List[Union[Agent, ConspiracyAgent]]:
        return self.schedule.agents


if __name__ == '__main__':
    model = ConspiracyModel(n_agents=10, width=10, height=10)
    model.step()
