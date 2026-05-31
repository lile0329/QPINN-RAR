import pennylane as qml
import torch
import numpy as np
import torch.nn as nn


class DVQuantumLayer(nn.Module):
    def __init__(self, args, diff_method="backprop"):
        super().__init__()

        """
        Initialize the quantum layer with the given number of qubits and arguments.

        Args:
            num_qubits (int): Number of qubits in the quantum circuit.
            args (dict): Additional arguments for the quantum circuit (e.g., hyperparameters).
            diff_method (str): Differentiation method for the QNode (default: "backprop").
        """
        self.num_qubits = args["num_qubits"]
        self.num_quantum_layers = args["num_quantum_layers"]
        self.shots = args["shots"]
        self.q_ansatz = args["q_ansatz"]
        self.problem = args["problem"]
        self.encoding = args.get("encoding", "angle")

        if self.q_ansatz == "layered":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 4,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "alternate":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    (self.num_qubits * 4) - 4,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        elif self.q_ansatz == "cascade":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 3,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "farhi":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    (2 * self.num_qubits - 2),
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "sim_circ_15":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 2,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "cross_mesh":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    (4 * self.num_qubits) +  self.num_qubits *(self.num_qubits - 1),
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        elif self.q_ansatz == "lay":
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 3,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        else:
            self.params = None

        if not hasattr(self, "params") or self.params is None:
            raise ValueError(
                "Parameters are not initialized. Check the q_ansatz value."
            )
        self._initialize_weights()

        self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=None)
        # Completely disable caching to avoid "value too large" errors with large batches
        # Try both caching=0 and cache=False to ensure compatibility across PennyLane versions
        # Also disable cache at device level if possible
        if hasattr(self.dev, 'cache'):
            self.dev.cache = None
        # Create QNode with caching disabled - try False first, fallback to 0
        try:
            self.circuit = qml.QNode(
                self._quantum_circuit, self.dev, interface="torch", diff_method=diff_method, cache=False
            )
        except TypeError:
            # Fallback to caching=0 if cache=False is not supported
            self.circuit = qml.QNode(
                self._quantum_circuit, self.dev, interface="torch", diff_method=diff_method, caching=0
            )
        # Explicitly disable cache on the circuit to prevent execution interface from using cache
        if hasattr(self.circuit, 'cache'):
            self.circuit.cache = None
        # Set execution config to disable caching at execution level
        try:
            # Try to disable caching in execution config if available
            if hasattr(qml, 'config') and hasattr(qml.config, 'set_execution_config'):
                qml.config.set_execution_config(cache=False)
            # Also try to set default execution config
            if hasattr(qml, 'config') and hasattr(qml.config, 'defaults'):
                if isinstance(qml.config.defaults, dict):
                    qml.config.defaults['execution.cache'] = False
        except:
            pass
        # Set a threshold for automatic chunking to prevent cache errors proactively
        # Use smaller chunk size to avoid PennyLane cache overflow errors
        # This is especially important for RAR evaluation with large candidate sets
        self.chunk_size = 1000  # Process batches larger than this in chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.stack([self.circuit(sample) for sample in x]) # this line does the same but very slow since it requires running the quantum circuit for each input instance
        
        # Force disable cache before execution to prevent "value too large" errors
        # This must be done at multiple levels to ensure caching is completely disabled
        if hasattr(self.circuit, 'cache'):
            self.circuit.cache = None
        if hasattr(self.dev, 'cache'):
            self.dev.cache = None
        # Also try to clear any execution cache
        try:
            if hasattr(qml, 'execute') and hasattr(qml.execute, '__globals__'):
                # Try to access and clear execution cache if it exists
                pass
        except:
            pass
        
        # Proactively chunk large batches to avoid cache errors
        # This is especially important for RAR evaluation with large candidate points
        if x.shape[0] > self.chunk_size:
            # Process in chunks to avoid cache overflow
            results = []
            for i in range(0, x.shape[0], self.chunk_size):
                chunk = x[i:i+self.chunk_size]
                # Force disable cache for each chunk before processing
                if hasattr(self.circuit, 'cache'):
                    self.circuit.cache = None
                if hasattr(self.dev, 'cache'):
                    self.dev.cache = None
                # Execute with explicit cache disabling
                try:
                    chunk_result = self.circuit(chunk)
                except ValueError as e:
                    if "value too large" in str(e):
                        # If still fails, use even smaller chunks
                        if hasattr(self.circuit, 'cache'):
                            self.circuit.cache = None
                        if hasattr(self.dev, 'cache'):
                            self.dev.cache = None
                        # Use very small chunks as last resort
                        sub_chunk_size = 500
                        sub_results = []
                        for j in range(0, chunk.shape[0], sub_chunk_size):
                            sub_chunk = chunk[j:j+sub_chunk_size]
                            if hasattr(self.circuit, 'cache'):
                                self.circuit.cache = None
                            sub_results.append(self.circuit(sub_chunk))
                        chunk_result = torch.cat(sub_results, dim=0)
                    else:
                        raise
                results.append(chunk_result)
            return torch.cat(results, dim=0)
        else:
            # For normal-sized batches, process directly
            try:
                # Force disable cache before execution
                if hasattr(self.circuit, 'cache'):
                    self.circuit.cache = None
                if hasattr(self.dev, 'cache'):
                    self.dev.cache = None
                return self.circuit(x)   # vectorized input, no Python loop.. transpose in the main function or here is important
            except ValueError as e:
                if "value too large" in str(e):
                    # Fallback: if error still occurs, process in smaller chunks
                    # Also try to clear cache before retrying
                    if hasattr(self.circuit, 'cache'):
                        self.circuit.cache = None
                    if hasattr(self.dev, 'cache'):
                        self.dev.cache = None
                    chunk_size = 500  # Use smaller chunks as fallback
                    results = []
                    for i in range(0, x.shape[0], chunk_size):
                        chunk = x[i:i+chunk_size]
                        # Ensure cache is disabled for each chunk
                        if hasattr(self.circuit, 'cache'):
                            self.circuit.cache = None
                        if hasattr(self.dev, 'cache'):
                            self.dev.cache = None
                        results.append(self.circuit(chunk))
                    return torch.cat(results, dim=0)
                else:
                    raise

    def _quantum_circuit(self, x):
        if self.encoding == "amplitude":
            qml.templates.AmplitudeEmbedding(
                x, wires=range(self.num_qubits), normalize=True, pad_with=0.0
            )
        else:
            qml.templates.AngleEmbedding(x, wires=range(self.num_qubits), rotation="X")

        if self.q_ansatz == "layered":
            for layer in range(self.num_quantum_layers):
                self.layered(self.params[layer])

        elif self.q_ansatz == "alternate":
            for layer in range(self.num_quantum_layers):
                self.alternate(self.params[layer])
        elif self.q_ansatz == "cascade":
            for layer in range(self.num_quantum_layers):
                self.cascade(self.params[layer])
        elif self.q_ansatz == "farhi":
            for layer in range(self.num_quantum_layers):
                self.farhi_ansatz(self.params[layer])

        elif self.q_ansatz == "sim_circ_15":
            for layer in range(self.num_quantum_layers):
                self.create_sim_circuit_15(self.params[layer])

        elif self.q_ansatz == "cross_mesh":
            for layer in range(self.num_quantum_layers):
                self.create_cross_mesh(self.params[layer])

        elif self.q_ansatz == "lay":
            for layer in range(self.num_quantum_layers):
                self.lay(self.params[layer])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def _initialize_weights(self):
        """Apply Xavier initialization to all layers."""

        if self.q_ansatz == "farhi":
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, (2 * self.num_qubits - 2))
            )
        elif self.q_ansatz in ["sim_circ_15"]:
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, self.num_qubits * 2)
            )
        elif self.q_ansatz in [ "alternate"]:
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, (self.num_qubits * 4) - 4)
            )

        elif self.q_ansatz in ["layered"]:
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, (self.num_qubits * 4))
            )
        elif self.q_ansatz == "cascade":
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, self.num_qubits * 3)
            )
        elif self.q_ansatz == "cross_mesh":
            torch.nn.init.xavier_normal_(
                self.params.view(
                    self.num_quantum_layers, (4 * self.num_qubits) +  self.num_qubits *(self.num_qubits - 1)
                )
            )
        elif self.q_ansatz == "lay":
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, self.num_qubits * 3)
            )
        else:
            raise ValueError("Invalid q_ansatz value.", self.q_ansatz)

    def layered(self, params):
        """
        Creates a quantum circuit with num_layers * num_qubits parameters.

        Args:
            params (list or tensor): A flat list or tensor of parameters with length num_layers * num_qubits.
            num_qubits (int): The number of qubits in the circuit.
            num_layers (int): The number of layers in the circuit.

        Returns:
            None: Constructs the quantum circuit.
        """
        assert params is not None and len(params) == self.num_qubits * 4, (
            "The number of parameters must be equal to 4* num_qubits."
        )

        # track the parameter index
        param_idx = 0

        # print(f"{len(params)=}")

        # apply RZ and RX gates for each qubit in the layer
        for qubit_id in range(self.num_qubits):
            # print(f"layer: {layer} ,{param_idx=}")
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RX(params[param_idx], wires=qubit_id)
            param_idx += 1

        # qml.Barrier(wires=range(self.num_qubits)) ### barriers are used for clarity of drawing of the quantum circuit; however, they slow down computation and hinder optimization

        for qubit_id in range(self.num_qubits):
            qml.CNOT(wires=[qubit_id, (qubit_id + 1) % self.num_qubits])

        # qml.Barrier(wires=range(self.num_qubits))

        for qubit_id in range(self.num_qubits):
            # print(f"layer: {layer} ,{param_idx=}")
            qml.RX(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1

    def alternate(self, params):
        """
        Build a variational circuit with alternating thinly dressed CNOT gates.

        Args:
            params (list or np.ndarray): Parameters for the circuit. Should have a size of
                                        `num_layers * num_qubits * 4` (4 parameters per thinly dressed CNOT gate).
        """
        assert params is not None and len(params) == (self.num_qubits * 4) - 4, (
            "The number of parameters must be equal to  num_qubits * 4."
        )

        param_idx = 0  # Initialize the parameter index

        def build_tdcnot(ctrl, tgt):
            """Build a thinly dressed CNOT gate with the required parameters."""
            nonlocal param_idx  # Allow modification of the outer variable
            qml.RY(params[param_idx], wires=ctrl)
            param_idx += 1
            qml.RY(params[param_idx], wires=tgt)
            param_idx += 1
            qml.CNOT(wires=[ctrl, tgt])
            qml.RZ(params[param_idx], wires=ctrl)
            param_idx += 1
            qml.RZ(params[param_idx], wires=tgt)
            param_idx += 1

        # add layers of the ansatz
        for i in range(self.num_qubits - 1)[::2]:
            ctrl, tgt = i, ((i + 1) % self.num_qubits)
            build_tdcnot(ctrl, tgt)

        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))

        for i in range(self.num_qubits)[1::2]:
            ctrl, tgt = i, ((i + 1) % self.num_qubits)
            build_tdcnot(ctrl, tgt)

    def cascade(self, params):
        def add_rotations():
            param_counter = 0
            for i in range(0, self.num_qubits):
                qml.RX(params[param_counter], wires=i)
                param_counter += 1

            for i in range(0, self.num_qubits):
                qml.RZ(params[param_counter], wires=i)
                param_counter += 1

            # barrier after entanglement
            # qml.Barrier(wires=range(self.num_qubits))

        def add_entangling_gates():
            param_counter = 0
            qml.CRX(params[param_counter], wires=[self.num_qubits - 1, 0])
            param_counter += 1
            for i in reversed(range(1, self.num_qubits)):
                qml.CRX(params[param_counter], wires=[i - 1, i])
                param_counter += 1

        # add layers of the ansatz
        add_rotations()
        add_entangling_gates()

    def farhi_ansatz(self, params):
        param_counter = 0

        # ensure there are enough parameters for both sets of gates
        if len(params) != (2 * self.num_qubits - 2):
            raise ValueError("Insufficient parameters for RXX and RZX gates")

        # custom RXX and RZX gate definitions
        def RXX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RX(theta, wires=wires[0])
            qml.CNOT(wires=wires)

        def RZX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=wires)

        # RXX gates
        for i in range(self.num_qubits - 1):
            RXX(params[param_counter], wires=[self.num_qubits - 1, i])
            param_counter += 1

        # RZX gates
        for i in range(self.num_qubits - 1):
            RZX(params[param_counter], wires=[self.num_qubits - 1, i])
            param_counter += 1

    def create_sim_circuit_15(self, params):
        """
        Creates a variational circuit based on circuit 15 in arXiv:1905.10876.

        Args:
            n_data_qubits (int): Number of qubits in the circuit
            layers (int): Number of layers in the circuit
            sweeps_per_layer (int): Number of sweeps per layer
            activation_function (callable, optional): Activation function to apply between layers

        Returns:
            callable: A function that constructs the quantum circuit with given parameters
        """
        if params is None or len(params) != 2 * self.num_qubits:
            raise ValueError("Insufficient parameters for RXX and RZX gates")

        param_index = 0

        # apply rotations
        def apply_rotations():
            nonlocal param_index
            for i in range(self.num_qubits):
                qml.RY(params[param_index], wires=i)
                param_index += 1

        # apply entangling gates block 1
        def apply_entangling_block1():
            for i in reversed(range(self.num_qubits)):
                qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

        # apply entangling gates block 2
        def apply_entangling_block2():
            for i in range(self.num_qubits):
                control_qubit = (i + self.num_qubits - 1) % self.num_qubits
                target_qubit = (control_qubit + 3) % self.num_qubits
                qml.CNOT(wires=[control_qubit, target_qubit])

        # main circuit construction
        apply_rotations()
        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))
        apply_entangling_block1()
        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))
        apply_rotations()
        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))
        apply_entangling_block2()
        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))

    def create_cross_mesh(self, params):
        """
        Creates a generalized version of Circuit 5 with CRZ gates where control and target wires
        are properly separated.

        Args:
            params (np.ndarray): Array of parameters for the rotation gates
        """
        param_idx = 0
        # verify parameter count
        expected_params = (4 * self.num_qubits) +  self.num_qubits *(self.num_qubits - 1)

        if params is None or len(params) != expected_params:
            raise ValueError(
                f"Expected {expected_params} parameters but got {params.shape}"
            )

        # initial Rx gates on all qubits
        for i in range(self.num_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1

        for i in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1

        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))
        # additional Rz gates for all except last qubit
        for i in range(self.num_qubits - 1, -1, -1):
            for j in range(self.num_qubits - 1, -1, -1):
                if j != i:
                    qml.CRZ(params[param_idx], wires=[i, j])
                    param_idx += 1

        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))

        # middle layer (using RX instead of CNOTs as in original)
        for i in range(self.num_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1

        for i in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1

        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))

    def lay(self, params):
        """
        Creates a quantum circuit with the "lay" ansatz structure.
        
        Structure:
        - Each qubit: Rz, Ry, Rz rotations
        - First column of CNOTs: CNOT(0->1), CNOT(1->2), ..., CNOT(n-2->n-1)
        - Second column of CNOTs: CNOT(n-1->0), CNOT(0->n-1)
        
        Args:
            params (list or tensor): Parameters for the circuit. Should have size
                                    num_qubits * 3 (3 parameters per qubit: Rz, Ry, Rz).
        """
        assert params is not None and len(params) == self.num_qubits * 3, (
            f"The number of parameters must be equal to num_qubits * 3, got {len(params)}"
        )
        
        param_idx = 0
        
        # Apply Rz, Ry, Rz rotations for each qubit
        for qubit_id in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RY(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1
        
        # First column of CNOTs: CNOT(0->1), CNOT(1->2), ..., CNOT(n-2->n-1)
        for qubit_id in range(self.num_qubits - 1):
            qml.CNOT(wires=[qubit_id, qubit_id + 1])
        
        # Second column of CNOTs: CNOT(n-1->0), CNOT(0->n-1)
        if self.num_qubits >= 2:
            # CNOT from last qubit to first qubit
            qml.CNOT(wires=[self.num_qubits - 1, 0])

    def quantum_tanh_n_qubits(self, params, scale=1.0):
        """
        Enhanced nonlinear quantum tanh activation with cross-qubit interactions

        Args:
            scale (float): Scaling factor for the activation
            params (list): List of trainable parameters for the rotations
        """
        if self.num_qubits is None:
            raise ValueError("Wires cannot be None.")

        if params is None:
            # create parameters for both direct and cross interactions
            n_params = self.num_qubits * (self.num_qubits - 1) // 2
            params = [scale * np.pi / 2.0 * index for index in range(n_params)]

        # add nonlinear phase shifts
        for index in range(self.num_qubits):
            qml.PhaseShift(np.sin(params[index]) * np.pi, wires=index)
