import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
#from qiskit_machine_learning.primitives import Estimator
from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
from qiskit_aer.primitives import Estimator as AerEstimator
#from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit.library import ZZFeatureMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.circuit.library import raw_feature_vector

import os
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# -------------------------------
# 1) R3 Ansatz (3 rotations + entanglement)
# -------------------------------
from typing import List, Tuple, Literal

def build_r3_ansatz(
    num_qubits: int,
    depth: int = 2,
    name: str = "θ",
    entanglement: Literal["linear", "ring"] = "linear",
) -> Tuple[QuantumCircuit, List]:
    theta = ParameterVector(name, depth * num_qubits * 3)
    
    qc = QuantumCircuit(num_qubits, name=f"R3_{depth}L")
    k = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.rx(theta[k + 0], q)
            qc.ry(theta[k + 1], q)
            qc.rz(theta[k + 2], q)
            k += 3
        if entanglement == "linear":
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)
        elif entanglement == "ring":
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)
            if num_qubits > 2:
                qc.cx(num_qubits - 1, 0)
    return qc, list(theta)

def compress_256d_to_8qubits(p):  # p: (batch,256)
    """
    Map 256D probability vector to 8 qubit Z-expectation values.
    Each qubit's Z expectation is sum over probabilities of 256 states,
    weighted by +1/-1 depending on the qubit being 0 or 1 in that state.
    """
    batch = p.shape[0]

    # Build z_matrix: 256 x 8
    # Each row = one computational basis state
    # Each column = qubit Z value (+1 for |0>, -1 for |1>)
    z_matrix = torch.zeros(256, 8, dtype=p.dtype, device=p.device)
    for i in range(256):
        bits = [(i >> (7-j)) & 1 for j in range(8)]  # MSB=qubit0
        z_matrix[i, :] = torch.tensor([1 if b==0 else -1 for b in bits], dtype=p.dtype)

    # Compute Z expectations: batch matmul p @ z_matrix
    # p shape: (batch,256), z_matrix: (256,8)
    z_exp = p @ z_matrix  # shape: (batch,8)

    # Normalize by sum of probabilities to ensure Z in [-1,1]
    z_exp = z_exp / p.sum(dim=1, keepdim=True)

    return z_exp  # shape: (batch,8)

def decompress_8qubits_to_256(z_exp):
    """
    Approximate reconstruction of 256D probabilities from 8 qubit Z-expectations.
    z_exp: shape (batch,8), values in [-1,1]
    """
    batch = z_exp.shape[0]
    n_qubits = 8
    # Map Z expectation [-1,1] → p0 = probability qubit=0
    p0 = (1 + z_exp)/2  # shape (batch,8)
    p1 = 1 - p0         # probability qubit=1

    # Compute 256D probabilities assuming independent qubits
    probs = torch.zeros(batch, 2**n_qubits, device=z_exp.device)
    for i in range(2**n_qubits):
        bits = [(i >> (7-j)) & 1 for j in range(8)]  # MSB=qubit0
        prob = torch.ones(batch, device=z_exp.device)
        for q in range(n_qubits):
            prob *= p0[:,q] if bits[q]==0 else p1[:,q]
        probs[:,i] = prob
    return probs  # shape (batch,256)

# -------------------------------
# 3) Complex Leaky ReLU
# -------------------------------
class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.real_leaky_relu = nn.LeakyReLU(negative_slope)
        self.imag_leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, input):
        return torch.complex(
            self.real_leaky_relu(input.real),
            self.imag_leaky_relu(input.imag)
        )

# -------------------------------
# 4) PQC Autoencoder
# -------------------------------

class PQCAutoencoder(nn.Module):
    def __init__(self, 
                 data_qubits=8, 
                 trash_qubits=1, 
                 layers=[2,2,2],
                 model_type="bottleneck", # list['bottleneck','reverse_bottleneck']
                 entanglement = "linear", # list['linear', 'ring']
                 useactivation = False,
                 device='cpu'):
        super().__init__()
        self.data_qubits = data_qubits
        self.trash_qubits = trash_qubits
        self.total_qubits = data_qubits # + trash_qubits
        self.device = device
        if isinstance(layers, int):
            layers = [layers] * 3
        elif isinstance(layers, (list, tuple)):
            if len(layers) != 3:
                raise ValueError("layers must have exactly 3 integers")
        else:
            raise TypeError("layers must be an int or a list/tuple of 3 integers")
        layers1, layers2, layers3 = layers

        # define the training circuit 
        self.qc = QuantumCircuit(data_qubits)

        # ZZ feature map or AngleFeather Map or Amplitude 
        #self.fm = ZZFeatureMap(feature_dimension=data_qubits, reps=1, entanglement='linear')
        #self.fm = raw_feature_vector(2 ** (data_qubits)) # note raw_feature_vector pair with SamplerQNN for the back of 2**data_qubits probability to compare with inputs
        
        # Angle encoding: map each inputs dimension to a rotation
        x = ParameterVector("x", 2**(data_qubits)) # 2**(data_qubits) inputs
        for i in range(data_qubits):
            for j in range(i, 2**(data_qubits), data_qubits):
                self.qc.ry(x[j], i)

        # trainable layers
        self.U1, self.th1 = build_r3_ansatz(data_qubits, layers1, name='θ1', entanglement= entanglement)
        self.U2, self.th2 = build_r3_ansatz(data_qubits - trash_qubits, layers2, name='θ2', entanglement= entanglement)
        self.U3, self.th3 = build_r3_ansatz(data_qubits, layers3, name='θ3', entanglement= entanglement)
        self.weight_param_count = len(self.th1) + len(self.th2) + len(self.th3)
        self.weights = nn.Parameter(torch.randn(self.weight_param_count, dtype=torch.float))

        # Ansatz build
        self.ansatz = QuantumCircuit(data_qubits)
        self.ansatz.compose(self.U1, range(data_qubits), inplace=True)
        self.ansatz.barrier()
        self.ansatz.compose(self.U2, range(data_qubits-trash_qubits), inplace=True)
        self.ansatz.barrier()
        self.ansatz.compose(self.U3, range(data_qubits), inplace=True)
        self.ansatz.barrier()

        # full circuit
        #self.qc.compose(self.fm, qubits=range(data_qubits), inplace=True) # amplitude_featuremap
        self.qc.compose(self.ansatz, range(data_qubits), inplace=True) # PQC machine learning model

        # Create 2^n projectors as observables: |i><i|
        self.observables = []
        for i in range(2**data_qubits):
            proj = np.zeros((2**data_qubits, 2**data_qubits), dtype=complex)
            proj[i, i] = 1.0
            obs = SparsePauliOp.from_operator(Operator(proj))
            self.observables.append(obs)
        
        self.weight_params = self.ansatz.parameters # trainable parameters number
        self.data_params = x
        #self.data_params = self.fm.parameters # inputs num that connect to ansatz
        # where self.weight_params == self.data_params
        
        # QNN
        self.estimator = Estimator() 
        #self.estimator = AerEstimator()
        # self.estimator = Estimator(options={"shots": 1024})

        self.qnn = EstimatorQNN(
            circuit=self.qc,
            input_params=self.data_params,
            weight_params=self.weight_params,
            observables=self.observables, # this is the observable.
            input_gradients=True,
            estimator=self.estimator
        )

        # return 2**data_qubit probability easier
        #self.qnn = SamplerQNN(
        #    circuit = self.qc,
        #    input_params = self.data_params,
        #    weight_params = self.weight_params
        #    )
        self.qnn_torch = TorchConnector(self.qnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(B*T, D).to(torch.complex64)
        #self.weights = nn.Parameter(torch.randn(len(self.weight_params)))
        # 2) run QNN
        #y_list = []
        #for i in range(B*T):
        #    #xi = state_vectors[i].unsqueeze(0)  # keep batch dim
        #    xi = torch.tensor(x_flat[i,:], dtype=torch.complex64)
        #    #yi = self.qnn_torch(xi)
        #    yi = self.qnn.foward(xi)
        #    y_list.append(yi)
        #y = torch.stack(y_list, dim=0).reshape(B*T, -1).to(torch.complex64)
        #print('y shape', y.shape)
        ## 3) decode to classical
        #out_list = []
        #for i in range(B*T):
        #    out_list.append(y[i])
        #out = torch.stack(out_list, dim=0)
        #print('out shape:', out.shape)
        #print('x_flat shape:', x_flat.shape)
        #print('x shape:', x.shape)
       
        out = self.qnn_torch(x_flat)
        out = out.reshape(B, T, D)
        print(out.shape)
        return out

    # --- Save parameters ---
    def save_params(self, directory, filename_prefix="current"):
        os.makedirs(directory, exist_ok=True)
        params = torch.tensor([p.bound for p in (self.th1 + self.th2 + self.th3)])
        torch.save(params, os.path.join(directory, f"{filename_prefix}.pt"))

    # --- Load parameters ---
    def load_current_params(self, directory, filename_prefix="current"):
        path = os.path.join(directory, f"{filename_prefix}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parameter file not found: {path}")
        loaded_params = torch.load(path)
        # assign values back to ParameterVector
        all_params = self.th1 + self.th2 + self.th3
        if len(loaded_params) != len(all_params):
            raise ValueError("Loaded params length does not match circuit params")
        for i, param in enumerate(all_params):
            param.set_value(float(loaded_params[i]))

    # Optional: for best parameters tracking
    def update_best_params(self, directory, metric, filename_prefix="best"):
        self.save_params(directory, filename_prefix)


