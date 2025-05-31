# Snowflurry: An Open-Source Quantum Computing Library

Snowflurry is an open-source quantum computing library developed in Julia by Anyon Systems.  It allows you to build, simulate, and run quantum circuits. A related library, SnowflurryPlots, provides visualization of simulation results using bar charts.  Useful for exploring quantum computing, its features are described in the [documentation](link-to-documentation-here) and the installation guide is available on the [GitHub page](link-to-github-here).  Like the PennyLane library, Snowflurry can be used to run quantum circuits on the MonarQ quantum computer.


## Installation

The quantum computer simulator with Snowflurry is available on all clusters.  The Julia programming language must be loaded before accessing Snowflurry:

```bash
name@server ~]$ module load julia
```

The Julia programming interface is then called, and the Snowflurry quantum library is loaded (approximately 5-10 minutes):

```julia
julia> import Pkg
julia> Pkg.add(url="https://github.com/SnowflurrySDK/Snowflurry.jl", rev="main")
julia> Pkg.add(url="https://github.com/SnowflurrySDK/SnowflurryPlots.jl", rev="main")
julia> using Snowflurry
```

Quantum logic gates and commands are described in the [Snowflurry documentation](link-to-documentation-here).


## Use Case: Bell States

Bell states are maximally entangled two-qubit states. They are simple examples of superposition and entanglement. The Snowflurry library allows you to construct the first Bell state as follows:

```julia
julia> using Snowflurry
julia> circuit = QuantumCircuit(qubit_count=2);
julia> push!(circuit, hadamard(1));
julia> push!(circuit, control_x(1,2));
julia> print(circuit)
Quantum Circuit Object:
qubit_count: 2
q[1]:──H────*──
│
q[2]:───────X──
```

In this code, the Hadamard gate creates an equal superposition of |0⟩ and |1⟩ on the first qubit, while the CNOT gate (controlled X gate) creates entanglement between the two qubits.  This results in an equal superposition of states |00⟩ and |11⟩, the first Bell state. The `simulate` function simulates the exact state of the system:

```julia
julia> state = simulate(circuit)
julia> print(state)   
4-element Ket{ComplexF64}:
 0.7071067811865475 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.7071067811865475 + 0.0im
```

The `readout` operation specifies which qubits are measured. The `plot_histogram` function from the SnowflurryPlots library visualizes the results:

```julia
julia> using SnowflurryPlots
julia> push!(circuit, readout(1,1), readout(2,2))
julia> plot_histogram(circuit, 1000)
```

**(Remember to replace `link-to-documentation-here` and `link-to-github-here` with the actual links.)**
