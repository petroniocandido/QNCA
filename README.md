# QNCA - Quantum Neural Cellular Automata

Petrônio C.  L. Silva  <span itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0002-1202-2552" href="https://orcid.org/0000-0002-1202-2552" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a></span>

![Quantum Computing](https://img.shields.io/badge/Quantum_Computing-233045?style=for-the-badge&logo=quantumcomputing&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Qiskit](https://img.shields.io/badge/Qiskit-%236929C4.svg?style=for-the-badge&logo=Qiskit&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

In case you have any questions, do not hesitate in contact us using the following e-mail: petronio.candido@ifnmg.edu.br

- Our QNCA approach mix the power of Quantum Neural Networks(QNN) with the Quantum Cellular Automata (QCA), using Parameterized Quantum Circuits (PQC) and Variational Quantum Algorithms (VQA) to teach recursive patterns

## Cellular Automata (AC) and Neural Cellular Automata (NCA)

- The concept of a cellular automaton (CA) is based on observing complex systems, where the global behavior emerges from the interactions of simple and local state transitional rules. In general, a CA is composed by:
  - A set of cells, denoted as $\mathcal{C} = \{c_1, \ldots, c_i, \ldots, c_N\}$, consists of N cells, each characterized by an internal state $s_i^t$, which varies over time $t = \{1, \ldots, T\}$. The collection of all state values across the cells at the given time $t$ is represented as $s^t=\{s_1^t,s_2^t, \ldots,s_N^t\}$
  - A neighborhood function $\mathcal{N}: \mathcal{C} \rightarrow \{\mathcal{C}\}^m$ returns the list of the $m$ neighbors of each agent  $a_i$. The function $V$ models the topology of the space in which the agents are embedded, with $m$ fixed in the case of CA or variable in the case of graph CA.
  - A state transition function $\mathcal{S}: (c_i, s^t, \mathcal{N}) \rightarrow s_i^{t+1}$  updates the state of each agent ($c_i$ ) based on the current states of all cells ($s^t$) and the state of its neighboring cells ($\mathcal{N}(c_i)$).

- The state transition function $\mathcal{S}$ governs the behavior of individual cells at the local scale and determines the overall behavior of the CA at the global scale. Traditional CA systems typically use deterministic finite-state automata $\mathcal{S}$, but the literature explores various alternatives, such as probabilistic rules, fuzzy systems, and others. Particularly in simulations of complex natural phenomena, it is often necessary for $\mathcal{S}$ to be learned from data that represents the underlying dynamics of these phenomena. Based on the literature, various approaches for training cell behavior have been proposed, including genetic algorithms, Fuzzy Time Series, and Neural Networks.
- Neural Cellular Automata (NCA) is a category of CA in which $\mathcal{S}$ is modeled by an artificial neural network. NCA extend the capabilities of traditional CA, enabling the representation of more complex local behaviors. NCA have been proposed for various applications, including feature extraction and image generation, among others.

## Quantum NCA

- The circuit architecture resembles the Quantum Convolutional Networks (QCNN), in the way it iteratively employs shared parameters in the unitary evolution of neighbor qubits.
- However key differences must be pointed: a) QCNN changes the qubits in place, b) each iteration
- Our implementation utilizes $2n$ qubits for a $n$ cells unidimensional grid. Each cell of the grid is represented by a qubit $|c_i⟩$, and for each cell qubit there is an ancilla qubit $|c_{i+n}⟩$. The $n$ cell qubits subset will contain the automata state in time $t$, where the other $n$ ancilla qubits subset will contain the automata state in time $t+1$. After each iteration the values of cell and ancilla quibts are swapped and the ancilla quibts are reset.
- This layout avoids evolving the state of the cell qubit $i$ from time $t$ to time $t+1$, when their state in time $t$ still needs to be used for evolving the states of the neighbor cells $i-1$ and $i+1$.
- While using ancilla quibts increase the width of the circuit, it grants the correct evolution of the cellular automaton.

- the length of the circuit is directed affected by the the number of iterations T, which imply in the number of repetitions of the unitary .

- Learning the shared circuit parameters is a challenge task due the iterative and recursive nature of the Cellular Automata. An annology for this complexity is to think that the same parameters are repeated verticully (among each set of neighbor qubits) and repated horizontally, for each iteration of the algorithm.

- For an $n$-grid 1D cellular automata with 2-cell neighborhood and T iterations, the $\theta$ will contains a fixed number of 18 parameters, but the circuit will contain $2n$ qubits of width and length of $O(9nT)$.



