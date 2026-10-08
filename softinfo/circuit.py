from dataclasses import dataclass

import stim


@dataclass(frozen=True)
class Noise:
    """Uniform circuit-level noise (paper App. C)."""
    p2: float = 0.0      # depolarizing after each CX
    p1: float = 0.0      # depolarizing after H, X, Z
    t1: float = 0.0      # X and Y error per round on code qubits (Eq. C1)
    t2: float = 0.0      # Z error per round on code qubits (Eq. C2)
    p_hard: float = 0.0  # X flip before each measurement
    p_soft: float = 0.0  # flip of the recorded bit only

    @staticmethod
    def from_list(noise_list):
        """From the original [p2, p1, t1, t2, p_readout, p_hard, p_soft] convention."""
        p2, p1, t1, t2, _, ph, ps = noise_list
        return Noise(p2, p1, t1, t2, ph, ps)


def repetition_code(d, T, xbasis=False, logical=0, noise=Noise(), subsampling=False):
    """Distance-d memory, T rounds, ancillas never reset (paper Fig. 1).

    Qubits 0..2d-2 on a line (code even, ancillas odd). Measurements: T rounds of the d-1 ancillas,
    then the d code qubits. Detectors D_t = m_t xor m_{t-2}, so a soft flip gives a distance-two time
    edge. `subsampling` adds the CX error boundary qubits pick up from the rest of the chain (App. E).
    """
    code, anc, n = list(range(0, 2 * d - 1, 2)), list(range(1, 2 * d - 1, 2)), d - 1
    c = stim.Circuit()

    def add(name, qubits, p=0.0):
        if p:
            c.append(name, qubits, p)

    def gate1(name, qubits):
        c.append(name, qubits)
        add("DEPOLARIZE1", qubits, noise.p1)

    def measure(qubits):
        add("X_ERROR", qubits, noise.p_hard)
        c.append("M", qubits, [noise.p_soft] if noise.p_soft else [])

    c.append("R", code + anc)
    if xbasis:
        gate1("H", code)
    if logical:
        gate1("Z" if xbasis else "X", code)
    for t in range(T):
        add("X_ERROR", code, noise.t1)
        add("Y_ERROR", code, noise.t1)
        add("Z_ERROR", code, noise.t2)
        if xbasis:
            gate1("H", anc)
        for layer, edge in ((code[:-1], code[-1]), (code[1:], code[0])):
            pairs = [q for a, b in zip(layer, anc) for q in ((b, a) if xbasis else (a, b))]
            c.append("CX", pairs)
            add("DEPOLARIZE2", pairs, noise.p2)
            add("DEPOLARIZE1", [edge], noise.p2 * subsampling)
            c.append("TICK")
        if xbasis:
            gate1("H", anc)
        measure(anc)
        for a in range(n):
            c.append("DETECTOR", [stim.target_rec(-n + a)] + [stim.target_rec(-3 * n + a)] * (t >= 2))
    if xbasis:
        c.append("H", code)
    measure(code)
    for a in range(n):
        c.append("DETECTOR", [stim.target_rec(r) for r in (-d + a, -d + a + 1, -d - n + a, -d - 2 * n + a)])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return c
