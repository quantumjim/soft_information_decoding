from qiskit_qec.circuits.repetition_code import ArcCircuit
from copy import copy
import pickle
from sklearn.mixture import GaussianMixture
from random import random
import numpy as np
from matplotlib import pyplot as plt
from qiskit_aer.noise import NoiseModel,pauli_error,depolarizing_error
from qiskit_qec.noise.paulinoisemodel import PauliNoiseModel
import sys
import time

def overlap(link1, link2):
    """Determine whether two links overlap"""
    return set(link1).intersection(link2) != set()

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    start = time.time()
    def show(j):
        x = int(size*j/count)
        remaining = ((time.time() - start) / j) * (count - j)
        
        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"
        
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
        
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def get_aer_noise_model(p):
    p_1, p_2, p_m = 0,p,p
    e_1 = depolarizing_error(p_1, 1)
    e_2 = depolarizing_error(p_2, 2)
    e_m = pauli_error([('X',p_m),('I',1-p_m)])
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(e_1, ['h', 'x', 'rz', 'sx'])
    noise_model.add_all_qubit_quantum_error(e_2, ['cx'])
    noise_model.add_all_qubit_quantum_error(e_2, ['cz'])
    noise_model.add_all_qubit_quantum_error(e_2, ['ecr'])
    noise_model.add_all_qubit_quantum_error(e_m, ['measure'])
    return noise_model

def get_qec_noise_model(p1Q:float = 0, p2Q:float = 0, pRR:float = 0):
    """
        Args:
            p1Q: single-qubit gate error probability; depolarizing channel with p1Q/3 for each Pauli: X, Y, Z
            p2Q: two-qubit gate error probability; 2Q depolarizing channel with p2Q/15 for each two-qubit Pauli that is not the identity
            pRR: error probability of readout (X error before measurement) and reset (X error after reset) errors
    """
    error_dict = {
        'reset': {
            "chan": {
                        'i':1-pRR,
                        'x':pRR
                    }
                },
        'measure': {
            "chan": {
                        'i':1-pRR,
                        'x':pRR
                    }
                },
        'h': {
            "chan": {
                        'i':1-p1Q
                    }|
                    {
                        i:p1Q/3
                        for i in 'xyz'
                    }
                },
        'cx': {
            "chan": {
                        'ii':1-p2Q
                    }|
                    {
                        i+j:p2Q/15
                        for i in 'ixyz' 
                        for j in 'ixyz' 
                        if i+j!='ii'
                    }
                },
                }
    noise_model = PauliNoiseModel(fromdict=error_dict)
    return noise_model


# figures out links and schedule for a given heavy hex device
def schedule_heavy_hex(backend, blacklist=[], rounds=["xz", "yx", "zy"]):

    try:
        raw_coupling_map = backend.configuration().coupling_map
        n = backend.configuration().num_qubits
        faulty_qubits = backend.properties().faulty_qubits()
        faulty_gates = backend.properties().faulty_gates()
    except:
        raw_coupling_map = backend.coupling_map
        n = backend.num_qubits
        faulty_qubits = []
        faulty_gates = []

    # remove any double counted pairs in the coupling map
    coupling_map = []
    for pair in raw_coupling_map:
        pair = list(pair)
        pair.sort()
        if pair not in coupling_map:
            coupling_map.append(pair)

    # find the degree for each qubit
    degree = [0] * n
    for pair in coupling_map:
        for j in range(2):
            degree[pair[j]] += 1
    degree = [int(deg) for deg in degree]

    # bicolor the qubits
    color = [None] * n
    color[0] = 0
    k = 0
    while None in color:
        for pair in coupling_map:
            for j in range(2):
                if color[pair[j]] is not None:
                    color[pair[(j + 1) % 2]] = (color[pair[j]] + 1) % 2
    # determine the color of vertex qubits
    for q in range(n):
        if degree[q] == 3:
            vertex_color = color[q]
            break
    # find  vertex qubits for each auxilliary
    link_dict = {}
    for q in range(n):
        if color[q] != vertex_color:
            link_dict[q] = []
    link_list = list(link_dict.keys())
    for pair in coupling_map:
        for j in range(2):
            if pair[j] in link_list:
                q = pair[(j + 1) % 2]
                if q not in link_dict[pair[j]]:
                    link_dict[pair[j]].append(q)
    # create the links list
    links = []
    for a, v0v1 in link_dict.items():
        if len(v0v1) == 2:
            links.append((v0v1[0], a, v0v1[1]))

    # find the plaquettes
    plaquettes = []
    all_paths = {}
    links_in_plaquettes = set({})
    for link in links:
        paths = [[[link]]]
        for l in range(6):
            paths.append([])
            for path in paths[l]:
                last_link = path[-1]
                for next_link in links:
                    if next_link != last_link:
                        if overlap(next_link, last_link):
                            try:
                                turn_back = overlap(next_link, path[-2])
                            except:
                                turn_back = False
                            if not turn_back:
                                if (next_link not in path) or l == 5:
                                    paths[-1].append(path.copy() + [next_link])
        for path in paths[6]:
            if path[0] == path[-1]:
                plaquette = set(path[:6])
                if plaquette not in plaquettes:
                    plaquettes.append(plaquette)
        all_paths[link] = paths

    # find the plaquettes neighbouring each link
    wings = {link: [] for link in links}
    for p, plaquette in enumerate(plaquettes):
        for link in plaquette:
            wings[link].append(p)

    # now assign a type (x, y or z) to each link so that none overlap
    link_type = {link: None for link in links}
    for unwinged in [False, True]:
        for r in ["x", "y", "z"]:
            # assign a single unassigned link as the current type
            for link in link_type:
                if link_type[link] is None and len(wings[link]) == 2:
                    link_type[link] = r
                    break

            # assign links that are 3 away in the plaquette or 2 away in different plaquettes as the same type
            all_done = False
            k = 0
            while all_done == False:
                newly_assigned = 0
                for l in [2, 3]:
                    for first in all_paths:
                        for path in all_paths[first][l]:
                            last = path[-1]
                            share_plaquette = False
                            for plaquette in plaquettes:
                                if first in plaquette and last in plaquette:
                                    share_plaquette = True
                            bulk = len(wings[first]) == 2 and len(wings[last]) == 2
                            if share_plaquette == (l == 3):
                                if l == 3 or bulk:
                                    link_pair = [first, last]
                                    for j in range(2):
                                        if (
                                            link_type[link_pair[j]] is not None
                                            and link_type[link_pair[(j + 1) % 2]]
                                            is None
                                        ):
                                            link_type[
                                                link_pair[(j + 1) % 2]
                                            ] = link_type[link_pair[j]]
                                            newly_assigned += 1
                all_done = newly_assigned == 0

        # if plaquettes have a single type missing, fill them in
        for plaquette in plaquettes:
            types = [link_type[link] for link in plaquette]
            for (r1, r2, r3) in [("x", "y", "z"), ("x", "z", "y"), ("z", "y", "x")]:
                if r1 in types and r2 in types and r3 not in types:
                    for link in plaquette:
                        if link_type[link] is None:
                            link_type[link] = r3

    # restrict `links` to only links with a type
    links = [link for link, r in link_type.items() if r]

    # bicolour the vertices
    vcolor = {links[0][0]: 0}
    for link in links:
        for j, k in [(0, -1), (-1, 0)]:
            if link[j] in vcolor and link[k] not in vcolor:
                vcolor[link[k]] = (vcolor[link[j]] + 1) % 2

    # find links around each vertex
    triplets = {v: {} for v in vcolor}
    for link, r in link_type.items():
        if link in links:
            for j in [0, -1]:
                if link[j] in triplets:
                    assert r not in triplets[link[j]]
                    triplets[link[j]][r] = link

    # schedule the entangling gates
    schedule = []
    for rr in rounds:
        round_schedule = []
        for v in triplets:
            r = rr[vcolor[v]]
            if r in triplets[v]:
                link = triplets[v][r]
                round_schedule.append((v, link[1]))
        schedule.append(round_schedule)

    # determine which pairs are blacklisted
    blacklist = set(blacklist + faulty_qubits)
    blacklisted_pairs = [set(g.qubits) for g in faulty_gates if len(g.qubits) > 1]
    for pair in raw_coupling_map:
        pair = set(pair)
        if pair not in blacklisted_pairs:
            if pair.intersection(blacklist):
                blacklisted_pairs.append(pair)

    # remove links with a blacklisted pair,
    # and blacklist the other pair in the link
    working_links = []
    for link in links:
        if (
            set(link[0:2]) not in blacklisted_pairs
            and set(link[1:3]) not in blacklisted_pairs
        ):
            working_links.append(link)
        else:
            if set(link[0:2]) in blacklisted_pairs:
                blacklisted_pairs.append(set(link[1:3]))
            if set(link[1:3]) in blacklisted_pairs:
                blacklisted_pairs.append(set(link[0:2]))
    links = working_links

    # remove corresponding gates from the schedule
    working_schedule = []
    for layer in schedule:
        working_layer = []
        for pair in layer:
            if set(pair) not in blacklisted_pairs:
                working_layer.append(pair)
        working_schedule.append(working_layer)
    schedule = working_schedule

    # check that it all worked
    num_cnots = 0
    cxs = []
    for round_schedule in schedule:
        num_cnots += len(round_schedule)
        cxs += round_schedule
        round_list = []
        for pair in round_schedule:
            round_list += list(pair)
        assert len(round_list) == len(set(round_list)), (
            len(round_list),
            len(set(round_list)),
        )
    assert num_cnots == len(cxs)
    for link in links:
        for pair in [tuple(link[0:2]), tuple(link[1:3])]:
            if pair not in cxs and pair[::-1] not in cxs:
                print(link)
    assert num_cnots == 2 * len(links), (num_cnots, 2 * len(links))

    return links, schedule, triplets, vcolor

def linearize_ARC(line, code):
    '''
    Given a list of qubits along a line across an ARC and the ARC itself,
    generate a 1D sub-ARC
    '''

    # links for linear code
    line_links = []
    for j in range(0,len(line)-2,2):
        line_links.append((line[j], line[j+1], line[j+2]))

    # cnots in linear code
    line_cnots = set()
    for link in line_links:
        line_cnots.add((link[0], link[1]))
        line_cnots.add((link[2], link[1]))

    # schedule of original code, including only cnots for linear code
    line_schedule = []
    for layer in code.schedule:
        line_layer = []
        for cnot in layer:
            if cnot in line_cnots:
                line_layer.append(cnot)
        line_schedule.append(line_layer)


    # make a function to convert original strings to 1D strings
    def linearize_string(line_code, string):
        string = string.split(' ')
        final = ['' for _ in range(line_code.num_qubits[0])]
        for q, lj in line_code.code_index.items():
            j = line_code.original_code.code_index[q]
            final[-lj-1] = string[0][-j-1]
        rounds = [['' for _ in range(line_code.num_qubits[1])] for _ in range(line_code.T)]
        for t in range(line_code.T):
            for q, lj in line_code.link_index.items():
                j = line_code.original_code.link_index[q]
                rounds[t][-lj-1] = string[t+1][-j-1]
            rounds[t] = ''.join(rounds[t])
        return ''.join(final) + ' ' + ' '.join(rounds)
    # monkey patch it in
    ArcCircuit.linearize_string = linearize_string

    # make a function to convert original nodes to 1D NODES
    def linearize_nodes(self, nodes):
        line_nodes = []
        for node in nodes:
            if tuple(node.qubits) in self.link_lookup:
                lnode = copy(node)
                l0, l1, idx = self.link_lookup[tuple(node.qubits)]
                lnode.qubits = [l0,l1]
                lnode.index = idx
                line_nodes.append(lnode)
        return line_nodes
    # monkey patch it in
    ArcCircuit.linearize_nodes = linearize_nodes

    # make the linear ARC
    line_code = ArcCircuit(
        line_links,
        code.T,
        schedule=line_schedule,
        run_202=code.run_202,
        basis=code.basis,
        logical=code.logical,
        resets=code.resets
    )
    # add original code as attribute (required by linearize_string)
    line_code.original_code = code
    # add a way to look up link indices as an attribute (required by linearize_nodes)
    link_lookup = {}
    for l, link in enumerate(line_links):
        link_lookup[link[0], link[2]] = (link[0], link[2],len(line_links)-l-1)
        link_lookup[link[2], link[0]] = (link[0], link[2],len(line_links)-l-1)
    line_code.link_lookup = link_lookup

    return line_code

class ArcParams:
    def __init__(
        self,
        links: list,
        T: int,
        basis: str = "xy",
        logical: str = "0",
        resets: bool = True,
        delay: int = None,
        barriers: bool = True,
        color: dict = None,
        max_dist: int = 2,
        schedule: list = None,
        run_202: bool = True,
        rounds_per_202: int = 9,
        conditional_reset: bool = False,
        note: str = '',
        job_ids: list = []
    ):
        self.links = links
        self.T = T
        self.basis = basis
        self.logical = logical
        self.resets = resets
        self.delay = delay
        self.barriers = barriers
        self.color = color
        self.max_dist = max_dist
        self.schedule = schedule
        self.run_202 = run_202
        self.rounds_per_202 = rounds_per_202
        self.conditional_reset = conditional_reset
        self.note = note
        self.job_ids = job_ids

class MemoryWrangler():
    def __init__(self, code, memory):
        self.code = code['0']
        self.memory = memory
        self.shots = len(memory[0])

        self.center = {}
        self.r = {}
        for q in self.code.qubits[0] + self.code.qubits[1]:
            self._get_circles_q(q)

    def get_point(self, j, s, q, t=0):
        '''
        Get the IQ plane point for the given parameters.
        
        Args:
            j: index of circuit
            s: shot
            q: qubit
            t: syndrome measurment round (for j!= 4 or 5)  

        Returns:
            x,y: real and imaginary parts of the point 
        '''
        iq_list = self.memory[j]
        if j in range(4):
            if q in self.code.code_index:
                k = - self.code.num_qubits[0] + self.code.code_index[q]
            elif q in self.code.link_index:
                k = t*self.code.num_qubits[1] + self.code.link_index[q]
        else:
            k = q
        x = iq_list[s][k].real
        y = iq_list[s][k].imag
        return x,y
    
    def _get_circles_q(self, q, plot=False):
        """
        Looks at the results of the 0 and 1 circuits for the
        given qubit. Determines the center points for the 0 and 1 clouds, as well as
        their radii (which is the std).
        """

        if plot:
            _, ax = plt.subplots()
            plt.gca().set_aspect("equal")

        points = [[], []]
        for j in [4, 5]:
            xs, ys = [], []
            for s in range(self.shots):
                x, y = self.get_point(j, s, q)
                xs.append(x)
                ys.append(y)
                points[j - 4].append((x, y))
            if plot:
                plt.scatter(xs, ys, s=1)

        if q not in self.center:
            model = GaussianMixture(n_components=2, covariance_type="full")
            pred = model.fit_predict(points[0] + points[1])
            if model.n_features_in_ < 1:
                print("Only one  cluster for qubit ", q)

            counts = [{0: 0, 1: 0, 2: 0} for _ in range(2)]
            for j in range(self.shots):
                counts[0][pred[j]] += 1
                counts[1][pred[j + self.shots]] += 1

            r = []
            center = []
            for j in range(2):
                key = max(counts[j], key=counts[j].get)

                r0 = np.sqrt(max([model.covariances_[j][k, k] for k in range(2)]))
                r.append(r0)

                center.append(tuple(model.means_[key]))
            self.center[q] = center
            self.r[q] = r

        if plot:
            for j in range(2):
                circle = plt.Circle(
                    (self.center[q][j][0], self.center[q][j][1]),
                    3 * self.r[q][j],
                    fill=False,
                )
                ax.add_patch(circle)

    def plot_circles(self, q):
        self._get_circles_q(q, plot=True)

    def _get_cert(self, ds, rs):
        is_1 = int(min(ds)==ds[1])
        return (np.exp(-ds[is_1]**2)/rs[is_1]) / ( (np.exp(-ds[0]**2)/rs[0]) + (np.exp(-ds[1]**2)/rs[1]) )

    def get_bit_memory(self, j, method=None):
        if method == "simple":
            bit_memory = []
            bit_certainty = []
            for s in range(self.shots):
                rawstring = ""
                iqs = self.memory[j][s]
                for c in iqs:
                    rawstring = str(int(c.real>0)) + rawstring
                string = ""
                certainty = []
                for b, bit in enumerate(rawstring):
                    string += bit
                    certainty.append(1)
                    if b == self.code.num_qubits[0] - 1:
                        string += " "
                        certainty.append(None)
                    elif (
                        b > self.code.num_qubits[0]
                        and (b - self.code.num_qubits[0] + 1) % self.code.num_qubits[1]
                        == 0
                    ):
                        string += " "
                        certainty.append(None)
                bit_memory.append(string[0:-1])
                bit_certainty.append(certainty[0:-1])
        else:
            bit_memory = []
            bit_certainty = []
            for s in range(self.shots):
                # the syndrome measurement rounds
                string = []
                round_certs = []
                for t in range(self.code.T)[::-1]:
                    bits = [None] * self.code.num_qubits[1]
                    certs = [None] * self.code.num_qubits[1]
                    for q in self.code.qubits[1]:
                        xy = self.center[q]
                        r = self.r[q]
                        x, y = self.get_point(j, s, q, t)
                        ds = []
                        for l in range(2):
                            ds.append(
                                np.sqrt((xy[l][0] - x) ** 2 + (xy[l][1] - y) ** 2)
                                / r[l]
                            )
                        idx = -1 - self.code.link_index[q]
                        if ds[1] == ds[0]:
                            bits[idx] = str(int(random() < 0.5))
                            certs[idx] = 0.5
                        else:
                            bits[idx] = str(int(ds[1] < ds[0]))
                            certs[idx] = self._get_cert(ds, r)
                    string.append("".join(bits))
                    round_certs.append(certs)

                bits = [None] * self.code.num_qubits[0]
                final_certs = [None] * self.code.num_qubits[0]
                for q in self.code.qubits[0]:
                    r = self.r[q]
                    xy = self.center[q]
                    x, y = self.get_point(j, s, q, t)
                    ds = []
                    for l in range(2):
                        ds.append(
                            np.sqrt((xy[l][0] - x) ** 2 + (xy[l][1] - y) ** 2) / r[l]
                        )
                    idx = -1 - self.code.code_index[q]
                    if ds[1] == ds[0]:
                        bits[idx] = str(int(random() < 0.5))
                        final_certs[idx] = 0.5
                    else:
                        bits[idx] = str(int(ds[1] < ds[0]))
                        is_1 = int(min(ds)==ds[1])
                        final_certs[idx] = self._get_cert(ds, r)
                string = "".join(bits) + " " + " ".join(string)
                certainty = final_certs
                for certs in round_certs:
                    certainty += [None] + certs

                bit_memory.append(string)
                bit_certainty.append(certainty)

        return bit_memory, bit_certainty