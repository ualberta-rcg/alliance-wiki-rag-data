# Snowflurry

Snowflurry is an open-source quantum computing library developed in Julia by Anyon Systems. It allows you to build, simulate, and execute quantum circuits.  A related library, SnowflurryPlots, allows you to visualize simulation results in a bar chart.  Useful for exploring quantum computing, features are available in the [documentation](link-to-documentation-here) and the installation guide is available on the [GitHub](link-to-github-here) page.  Similar to the PennyLane library, Snowflurry can be used to run quantum circuits on the MonarQ quantum computer.


## Installation

The quantum computer simulator with Snowflurry is available on all our clusters. The Julia programming language must be loaded before accessing Snowflurry with the command:

```bash
[name@server ~]$ module load julia
```

Then, the Julia programming interface is called and the Snowflurry quantum library is loaded (approximately 5-10 minutes) with the commands:

```julia
[name@server ~]$ julia
julia> import Pkg
julia> Pkg.add(url="https://github.com/SnowflurrySDK/Snowflurry.jl", rev="main")
julia> Pkg.add(url="https://github.com/SnowflurrySDK/SnowflurryPlots.jl", rev="main")
julia> using Snowflurry
```

Quantum logic gates and commands are described in the [Snowflurry documentation](link-to-snowflurry-docs-here).


## Example Usage: Bell States

Bell states are maximally entangled two-qubit states. Two simple examples of quantum phenomena are superposition and entanglement. The Snowflurry library allows you to construct the first Bell state as follows:

```julia
[name@server ~]$ julia
julia> using Snowflurry
julia> circuit = QuantumCircuit(qubit_count=2);
julia> push!(circuit,hadamard(1));
julia> push!(circuit,control_x(1,2));
julia> print(circuit)
Quantum Circuit Object:
qubit_count: 2
q[1] :──H────*──
¦
q[2] :───────X──
```

In the code section above, the Hadamard gate creates an equal superposition of |0⟩ and |1⟩ on the first qubit while the CNOT gate (controlled X gate) creates entanglement between the two qubits.  An equal superposition of the states |00⟩ and |11⟩ is obtained, which is the first Bell state. The `simulate` function allows you to simulate the exact state of the system.

```julia
julia> state = simulate(circuit)
julia> print(state)   
4-element Ket{ComplexF64}:
 0.7071067811865475 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.7071067811865475 + 0.0im
```

To take a measurement, the `readout` operation allows you to specify which qubits will be measured. The `plot_histogram` function from the SnowflurryPlots library allows you to visualize the results.

```julia
[name@server ~]$ julia
julia> using SnowflurryPlots
julia> push!(circuit, readout(1,1), readout(2,2))
julia> plot_histogram(circuit,1000)
```


**(Remember to replace the placeholder links with the actual links.)**
