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
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp
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

# -------------------------------
# 2) Amplitude Encode / Decode
# -------------------------------
class AmplitudeFeatureMap(nn.Module):
    """Normalize classical vector into quantum amplitudes"""
    def __init__(self, num_qubits: int = 8, eps: float = 1e-12):
        super().__init__()
        self.num_qubits = num_qubits
        self.dim = 2 ** self.num_qubits
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2**num_qubits)
        #psi = x.to(torch.complex64)
        psi = torch.tensor(x, dtype=torch.complex64)
        if psi.ndim == 1:
            psi=psi.unsqueeze(0)
        norm = torch.linalg.norm(psi, dim=1, keepdim=True)
        norm = torch.where(norm < self.eps, torch.tensor(1.0, device=x.device, dtype=torch.complex64), norm)
        return psi / norm

class AmplitudeDecode(nn.Module):
    """Decode n-qubit state vector to classical probabilities"""
    def __init__(self, input_dim: int = 8, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, 2**input_dim) complex tensor
        norm = torch.linalg.norm(y, dim=1, keepdim=True)
        y_normalized = y / torch.where(norm==0, torch.tensor(1e-8, device=y.device, dtype=y.dtype), norm)
        classical_vector = y_normalized.abs() ** 2
        if classical_vector.shape[1] < self.output_dim:
            padding = torch.zeros(classical_vector.shape[0], self.output_dim - classical_vector.shape[1], device=y.device)
            classical_vector = torch.cat([classical_vector, padding], dim=1)
        else:
            classical_vector = classical_vector[:, :self.output_dim]
        return classical_vector

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
        
        #print(type(layers), layers)
        # amplitude encode / decode
        self.amplitude_encode = AmplitudeFeatureMap(data_qubits)
        self.amplitude_decode = AmplitudeDecode(data_qubits, 2**data_qubits)

        # ZZ feature map or AngleFeather Map 
        #self.fm = ZZFeatureMap(feature_dimension=data_qubits, reps=1, entanglement='linear')
        self.fm = raw_feature_vector(2 ** (data_qubits))

        if isinstance(layers, int):
            layers = [layers] * 3
        elif isinstance(layers, (list, tuple)):
            if len(layers) != 3:
                raise ValueError("layers must have exactly 3 integers")
        else:
            raise TypeError("layers must be an int or a list/tuple of 3 integers")
        
        layers1, layers2, layers3 = layers

        # Ansatz parameterization:  build 3-layer R3 ansatz
        self.U1, self.th1 = build_r3_ansatz(data_qubits, layers1, name='θ1', entanglement= entanglement)
        self.U2, self.th2 = build_r3_ansatz(data_qubits - trash_qubits, layers2, name='θ2', entanglement= entanglement)
        self.U3, self.th3 = build_r3_ansatz(data_qubits, layers3, name='θ3', entanglement= entanglement)
        
        # Ansatz build
        self.ansatz = QuantumCircuit(data_qubits)
        self.ansatz.compose(self.U1, range(data_qubits), inplace=True)
        self.ansatz.barrier()
        self.ansatz.compose(self.U2, range(data_qubits-trash_qubits), inplace=True)
        self.ansatz.barrier()
        self.ansatz.compose(self.U3, range(data_qubits), inplace=True)
        self.ansatz.barrier()

        # full circuit
        self.qc = QuantumCircuit(self.data_qubits)
        self.qc.compose(self.fm, qubits=range(data_qubits), inplace=True) # amplitude_featuremap
        self.qc.compose(self.ansatz, range(data_qubits), inplace=True) # PQC machine learning model

        # create Z observable on each data qubit
        self.observables = [
        SparsePauliOp("".join(['Z' if i == j else 'I' for i in range(self.data_qubits)]))
            for j in range(self.data_qubits)
        ]

        self.weight_params = self.ansatz.parameters # trainable parameters number
        self.data_params = self.fm.parameters # inputs num that connect to ansatz
        # where self.weight_params == self.data_params

        # QNN
        self.estimator = Estimator() 
        #self.estimator = AerEstimator()
        # self.estimator = Estimator(options={"shots": 1024})

        self.qnn = EstimatorQNN(
            circuit=self.qc,
            input_params=self.data_params,
            weight_params=self.weight_params,
            observables=self.observables,
            input_gradients=True,
            estimator=self.estimator
        )
        self.qnn_torch = TorchConnector(self.qnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(B*T, D)

        # 1) Amplitude encode
        #state_circuits = [self.amplitude_encode(x_flat[i].cpu().numpy()) for i in range(B*T)]
        # TODO: each state_circuit can be run through QNN

        # 2) run QNN (simulator)
        #y_list = []
        #for i in range(B*T):
        #    xi = torch.tensor(state_circuits[i], dtype=torch.float32)
        #    yi = self.qnn_torch(xi.squeeze())
        #    y_list.append(yi)
        #y = torch.stack(y_list, dim=0)
        
        # 3) amplitude decode to classical vector
        #out_list = []
        #for i in range(B*T):
        #    out_list.append(self.amplitude_decode(y[i]))
        #out = torch.stack(out_list, dim=0).reshape(B, T, D)
        # 1) Amplitude encode (256 → 2**num_qubits)
        state_vectors = self.amplitude_encode(x_flat)

        # 2) run QNN
        y_list = []
        for i in range(B*T):
            #xi = state_vectors[i].unsqueeze(0)  # keep batch dim
            xi = torch.tensor(state_vectors[i], dtype=torch.complex64)
            yi = self.qnn_torch(xi)
            y_list.append(yi)
        y = torch.stack(y_list, dim=0).reshape(B*T, -1)

        # 3) decode to classical
        out_list = []
        for i in range(B*T):
            out_list.append(self.amplitude_decode(y[i].unsqueeze(0)))
        out = torch.stack(out_list, dim=0).reshape(B, T, D)
        return out
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


