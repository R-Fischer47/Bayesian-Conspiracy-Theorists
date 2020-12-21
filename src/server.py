import matplotlib.cm as cm
import matplotlib.colors as c
import mesa.visualization.modules as visualization
from mesa.visualization.ModularVisualization import ModularServer

from agent import ConspiracyAgent
from model import ConspiracyModel

grid_width = 30
grid_height = 30

model_params = {
    'n_agents': 200,
    'width': grid_width,
    'height': grid_height,

    'agent_reach_radius': 10,

    # Bayesian stuff
    'prior_sample_size': 5,
    'initial_sd': .25,
    'start_p_h': 0.15,
}


def draw_agent(agent: ConspiracyAgent) -> dict:
    portrayal = {
        'Shape': 'circle',
        'r': 0.5,
        'Filled': 'true',
        'Layer': 0,
        'Color': c.to_hex(cm.gnuplot2(agent.prior_value)),
    }

    return portrayal


visualization_elements = [
    visualization.CanvasGrid(
        portrayal_method=draw_agent,
        grid_width=grid_width,
        grid_height=grid_height,
        canvas_width=1000,
        canvas_height=1000,
    ),
]

server = ModularServer(
    ConspiracyModel,
    visualization_elements,
    "Bayesian Conspiracy Theorists",
    model_params,
)
