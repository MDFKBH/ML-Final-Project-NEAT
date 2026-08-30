# Vehicle Route Optimization — Machine Learning Final Project

A **fork of [NeuralNine/ai-car-simulation](https://github.com/NeuralNine/ai-car-simulation)**, used as the subject of our Machine Learning final project at Tunghai University (Fall 2024). The upstream repository supplies the pygame simulation and the `neat-python` integration; we studied that code, ran it across the six tracks, and adjusted its parameters until it solved the track the original settings could not.

**Presentation:** [*Vehicle Route Optimization* — final presentation slides](https://drive.google.com/drive/folders/1FK7mEH16lqAlgYZqKUG-sjISO-0odVCx?usp=sharing) (26 slides, Google Drive)

**Team:** 黃少鯤 · 黃睿廷 · 黃子修 · 鄭聿宏 · 蔡承赫

> The simulation, radar sensing, collision detection and NEAT integration are upstream code. Our changes are the parameter and fitness adjustments described in [§5 Analysis](#5-analysis), and are visible in the diff against the fork point:
> ```
> git diff ccb6509 HEAD
> ```

---

## 1. Introduction

Vehicle Route Optimization is the process of finding the most efficient routes. Traditional methods such as heuristic algorithms or mathematical optimization are often limited in their scalability and adaptability, whereas machine learning offers dynamic weights and data-driven solutions.

## 2. What is NEAT?

**NEAT** (NeuroEvolution of Augmenting Topologies) evolves both the weights *and* the structure of a neural network with a genetic algorithm.

**Reinforcement learning.** The network improves by self-evolution against a reward system, so no labelled dataset is required.

**Topology mutation.** Beyond adjusting weights dynamically, NEAT can create new nodes and generate new links, so the network structure grows over generations.

**Genome.** Each individual is described by node genes and connection genes.

**Speciation.** Similar networks are classified into species that compete independently, so a new structure is not immediately outcompeted before it has time to develop.

**Fitness sharing.** Fitness is shared within a species, which prevents any one species from monopolizing the population.

**Innovation numbers.** Every new gene receives a unique innovation number that tracks mutation history and makes crossover between different topologies possible.

## 3. Training Process

**Step 1 — Initialization.** Set the car's initial position, angle and speed. The radars are initialized to detect distances.

**Step 2 — Sensor input.** The radars detect the distance from the track borders at different angles. These distance values are the inputs to the neural network.

**Step 3 — Neural network control.** The network processes the radar inputs, and its outputs control the car's actions. The car moves according to those outputs.

**Step 4 — Fitness calculation.** Performance is measured by tracking the distance travelled. A car dies when it touches the track border, which is detected by the border colour. Each car in the generation receives a fitness score.

**Step 5 — Evolution.** NEAT evaluates the fitness of each car, and the better ones are selected for crossover and mutation, with the goal of evolving better-performing cars.

## 4. Project Demo

### Network

| | |
|---|---|
| **Input layer** | Radar 1 – Radar 5 |
| **Output layer** | Left · Right · Speed Up · Slow Down |

### Results with the original parameters

We ran `map1`–`map5` with the upstream settings (`pop_size = 30`, `num_outputs = 4`, `compatibility_threshold = 2`, `max_stagnation = 20`, `species_elitism = 2`, `survival_threshold = 0.2`). Terminal output for each run is in [`pic/`](pic).

| Track | Generation | Best fitness | Species |
|---|---|---|---|
| `map1` | 9 | 373,492 | 1 |
| `map2` | 14 | 310,676 | 1 |
| `map3` | 24 | 240,200 | 2 |
| `map4` | 19 | 288,500 | 2 |
| **`map5`** | **246** | **8,550** | 2 |

The first four tracks reached 240k–373k fitness within 24 generations. **`map5` failed**: after 246 generations the best fitness was 8,550, and the `stag` column shows the leading species had not improved for 222 generations.

![map5 with the original parameters: best fitness 8,550 at generation 246](pic/map5_fail.png)

<details>
<summary>Terminal output for map1–map4</summary>

| | |
|---|---|
| ![map1](pic/map1.png) | ![map2](pic/map2.png) |
| ![map3](pic/map3.png) | ![map4](pic/map4.png) |

</details>

### `map5` with the adjusted parameters

After the changes listed in §5, the same algorithm solved `map5`.

![map5 with the adjusted parameters: best fitness 1,080,300 at generation 69](pic/map5_solved.png)

| `map5` | Original parameters | Adjusted parameters |
|---|---|---|
| Generation shown | 246 | 69 |
| Best fitness | **8,550** | **1,080,300** |
| Population mean fitness | 2,558 | 113,717 |
| Population / species | 30 / 2 | 100 / 7 |
| Stagnation of the top species | 222 generations | — |
| Species at the best score | 1 of 2 | 4 of 7 |

*These are snapshots taken while each run was in progress, not converged final scores. Note that `pic/mapN.png` is the terminal output for the run on track `mapN.png`, not a picture of the track itself.*

## 5. Analysis

### `config.txt`

| Parameter | Original | Adjusted |
|---|---|---|
| `pop_size` | 30 | **100** |
| `num_outputs` | 4 | **2** |
| `compatibility_threshold` | 2 | **2.5** |
| `max_stagnation` | 20 | **30** |
| `species_elitism` | 2 | **3** |
| `survival_threshold` | 0.2 | **0.15** |

### `newcar.py`

**Fitness function** — a penalty for crashing was added:

```python
# Original                                  # Adjusted
def get_reward(self):                       def get_reward(self):
    return self.distance / (CAR_SIZE_X / 2)     if not self.alive:
                                                    return -100
                                                return self.distance / (CAR_SIZE_X / 2)
```

**Episode length** — the time limit per generation was extended:

```python
# Original                    # Adjusted
if counter == 60 * 20:        if counter == 60 * 160:
    break                         break
```

Other changes: the initial speed was lowered from 20 to 5, the display was switched from full screen to windowed, the track filename was lifted into a single `MAP` constant, and the HUD text was repositioned for the 1920×1080 window. `map6.png`, `requirements.txt` and `.gitignore` were added, and `map.png` was renamed to `map1.png`.

## 6. Conclusion

NEAT has the potential to play a significant role in solving highly complex and dynamic problems, especially in scenarios requiring automated exploration and optimization. It is poised to become a key foundational technology in the field of artificial intelligence, with applications in robotics, finance, games and self-driving.

---

## Running it

```bash
pip install -r requirements.txt
python newcar.py
```

Switch tracks by editing the `MAP` constant at the top of `newcar.py` (`map1.png` … `map6.png`). The window is 1920×1080.

## Follow-up work

This project was extended the following semester into a Deep Learning capstone that rebuilds the same environment as a Gymnasium environment trained with PPO, and runs a controlled ablation over reward and observation design: **[rl-car-reward-vs-observation](https://github.com/h-s-i-u/rl-car-reward-vs-observation)**.

## Credits

The simulation, radar sensing, collision detection and NEAT integration are the work of **[NeuralNine (Florian Dedov)](https://github.com/NeuralNine/ai-car-simulation)**, itself based on a tutorial by **Cheesy AI**. Full credit to them for the original project. This fork is for academic coursework only, with no commercial intent.

## References

1. https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies
2. https://macwha.medium.com/evolving-ais-using-a-neat-algorithm-2d154c623828
3. https://neat-python.readthedocs.io/en/latest/neat_overview.html
4. https://github.com/NeuralNine/ai-car-simulation
5. https://www.youtube.com/watch?v=Cy155O5R1Oo
6. https://towardsdatascience.com/how-do-we-teach-a-machine-to-program-itself-neat-learning-bb40c53a8aa6
7. https://drive.google.com/viewerng/viewer?url=http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
8. https://chatgpt.com/
9. https://www.youtube.com/watch?v=b3D8jPmcw-g&t=6s
10. https://www.youtube.com/watch?v=nPW6tKeapsM
11. https://github.com/ArztSamuel/Applying_EANNs?tab=readme-ov-file
12. https://arztsamuel.github.io/en/projects.html
13. https://www.youtube.com/watch?v=JpyVe9UU-w8&t=90s
14. https://www.youtube.com/watch?v=AJ1TR28KNqY
15. https://www.youtube.com/watch?v=2o-jMhXmmxA
16. https://www.youtube.com/watch?v=uEzhTYiQdh8
17. https://www.youtube.com/watch?v=qv6UVOQ0F44&t=44s
