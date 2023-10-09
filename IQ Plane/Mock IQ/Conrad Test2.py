# Checking probabilities result from Qiskit Aer
from qiskit import QuantumCircuit
import qiskit_aer as aer

N = 10
circ = QuantumCircuit(N)
circ.x(0)
# for x in range(N):
#     circ.h(x)
circ.save_probabilities()
circ.measure_all(inplace=True)

sim = aer.AerSimulator(shots=20000)
res = sim.run(circ)
print(res)
