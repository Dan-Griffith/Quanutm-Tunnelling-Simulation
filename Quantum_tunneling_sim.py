#Dan Griffith 
#Cole Mclaren 
#COMP 4900 A 

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, Diagonal
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math 

# --- Number of Qubits, Time step, Number of Time Steps ---
N_qubits = 4
delta_t = 0.01
steps = 1

# --- Well Constants ---
a = 50
b = 0.5
c= 0.9

# Double well 
positions = np.linspace(-1, 1, 2**N_qubits)
V = a * (positions**2 - b**2)**2

# --- Triple Well Configuration Using c ---
# V = a * (positions**2 - b**2)**2 * (positions**2 - c**2)

V = V.tolist()
initial_state = Statevector.from_label('00111110')

if math.log2(len(initial_state)) != N_qubits:
    raise ValueError(f"Length of initial states is {len(initial_state)} cannot be represented by {N_qubits} qubits.")

def gen_phi_pairs(N_qubits):
    phi_pairs = []
    line_operator = 0
    for i in range(1,N_qubits):
        current_line = N_qubits -i
    
        for i in range(N_qubits-1):
            line_operator = (N_qubits -2) - i
            pair = (current_line, line_operator)

            if not line_operator >= current_line:
                phi_pairs.append(pair)
 
    return phi_pairs

# --- Define Circuit ---
qc = QuantumCircuit(N_qubits)

# Step 1: QFT
qft = QFT(num_qubits=N_qubits)
qc.append(qft, range(N_qubits))

# Step 2: D gate (Z and Φ phases) - Tune Rotations to Specific Particle Attributes
theta_z = [np.pi**2 / (4 ** (N_qubits - k)) for k in range(N_qubits)]
phi_pairs = gen_phi_pairs(N_qubits)
theta_phi = [np.pi**2 / (2**i) for i in range(1, len(phi_pairs)+1)]
D_diag = []

for i in range(2 ** N_qubits):
   
    b = f"{i:0{N_qubits}b}"
    z_phase = sum(theta_z[k] for k in range(N_qubits) if b[(N_qubits-1) - k] == '1')
    phi_phase = sum(theta for (c, t), theta in zip(phi_pairs, theta_phi)
                    if b[(N_qubits-1)  - c] == '1' and b[(N_qubits-1)  - t] == '1')
    D_diag.append(np.exp(-1j * (z_phase + phi_phase) * delta_t))

diag = Diagonal(D_diag)
diag.name = 'D'
qc.append(diag, range(N_qubits))

# Step 3: Inverse QFT
qc.append(QFT(num_qubits=N_qubits, inverse=True, name='QFT^-1'), range(N_qubits))

# Step 4: Apply Well Potential
Q_diag = [np.exp(-1j * v * delta_t) for v in V]
diag_2 = Diagonal(Q_diag)
diag_2.name  = 'Q'
qc.append(diag_2, range(N_qubits))

# --- Simulate Evolution ---
probs_per_step = []
evolution = qc
for _ in range(steps):
    evolution = evolution.compose(qc)
    current_state = initial_state.evolve(evolution)

    # Calculate the probabilities (squared magnitude of the amplitudes)
    probs = np.abs(current_state.data) ** 2
    
    # Append the probabilities for this step
    probs_per_step.append(probs)

# --- Draw The Circuit ---
qc.draw(output='mpl')
maxes = np.max(probs_per_step, axis=1)

final_state = initial_state.evolve(evolution)
max_V = max(V)
V_scaled = [v / max_V for v in V]

# --- Plot Result ---
probs = np.abs(final_state.data) ** 2
states = [f"{i:0{N_qubits}b}" for i in range(2 ** N_qubits)]

plt.figure(figsize=(10, 6))
bars = plt.bar(states, probs, color='blue', alpha=0.7, label="Quantum Probability")

plt.plot(states, V_scaled, color='green', label="Scaled Potential Energy", linewidth=2)
plt.xlabel("Basis States (Position)")
plt.ylabel("Probability / Scaled Energy")
plt.title(f"Tunneling After {steps} Steps (from 1/√2|0011⟩ + 1/√2|1100⟩)")

plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.legend()
plt.tight_layout()
plt.show()

# --- Show Particle Animation ---
fig, ax = plt.subplots(figsize=(10, 4))

# Basis state positions
positions = np.arange(len(states))

ax.plot(positions, V_scaled, color='green', label='Scaled Potential Energy', linewidth=2)
max_line, = ax.plot([], [], 'ro', label='Particle', markersize=10)
counter_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, ha='left', va='top')

ax.set_xticks(positions)
ax.set_xticklabels(states, rotation=45)
ax.set_ylim(0, 1.1)
ax.set_xlim(-0.5, len(states) - 0.5)
ax.set_xlabel("Basis States (Position)")
ax.set_title("Quantum Particle Tunneling Over Potential Barrier")
ax.grid(True)
ax.legend()

def init():
    max_line.set_data([], [])
    counter_text.set_text('')
    return max_line, counter_text

def animate(i):
    max_index = np.argmax(probs_per_step[i])
    max_line.set_data([positions[max_index]], [0.6])  # Fixed vertical height
    counter_text.set_text(f"Step: {i}")
    return max_line, counter_text

# --- Run Animation ---
ani = FuncAnimation(
    fig, animate, frames=len(probs_per_step),
    init_func=init, blit=True, interval=100
)

plt.tight_layout()
plt.show()



