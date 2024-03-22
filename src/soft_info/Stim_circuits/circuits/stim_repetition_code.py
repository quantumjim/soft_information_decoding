# Maurice D. Hanisch
# 2023-03-20

import stim

class RepetitionCodeStimCircuit():
    def __init__(self,
                 d: int,
                 T: int,
                 xbasis: bool = False,
                 resets: bool = True,
                 noise_list: list = None,
                 subsampling: bool = False,
                 ):
        """
        Creates the STIM circuits corresponding to a logical 0 and 1 encoded 
        using a repetiton code.

        Args:
            d (int): Number of code qubits (and hence repetitions) used.
            T (int): Number of rounds of ancilla-assisted syndrome measurement.
            xbasis (bool): Whether to use the X basis to use for encoding (Z basis used by default).
            resets (bool): Whether to include a reset gate after mid-circuit measurements.
            noise_list (list): [twog_err, sglg_err, t1_err, t2_err, readout_err, hard_err, soft_err]
        """
        

        assert T > 0, "At least one round of syndrome measurement is required."
        # assert xbasis == False, "X basis not yet supported."
        # assert resets == True, "No Resets not yet fully supported."


        # Code parameters
        self.d = d
        self.T = T
        self._xbasis = xbasis
        self._resets = resets

        # Noise parameters
        if noise_list is None:
            noise_list = [0, 0, 0, 0, 0, 0, 0]
        self.twog_err = noise_list[0]
        self.sglg_err = noise_list[1]
        self.t1_err = noise_list[2]
        self.t2_err = noise_list[3]
        self.readout_err = noise_list[4]
        self.hard_err = noise_list[5]
        self.soft_err = noise_list[6]
        self.subsampling = subsampling

        # Qubit indices
        self.qubits = list(range(2*d-1))
        self.code_qubits = self.qubits[0::2]
        self.link_qubits = self.qubits[1::2]

        # Create the entanglement and detector blocks
        self._Z_ent_block = stim.Circuit()
        self._X_ent_block = stim.Circuit()
        self._det_block = stim.Circuit()
        self._create_blocks()

        # Create the circuits
        self.circuits = {}
        for log in ["0", "1"]:
            self.circuits[log] = stim.Circuit()

        # Generate the circuits
        self._preparation()
        self._first_round()
        if resets == False:
            self._first_round() # Repeat the first round for the detectors
        self._subsequent_rounds()
        self._final_readout()


    def _x(self, logs=("0", "1")):
        """Applies a logical X gate to the code qubits."""
        for log in logs:
            if self._xbasis:
                self.circuits[log].append('Z', self.code_qubits)
                self.circuits[log].append('DEPOLARIZE1', self.code_qubits, arg=self.sglg_err) if self.sglg_err > 0 else None
                self.circuits[log].append('TICK')
            else:
                self.circuits[log].append('X', self.code_qubits)
                self.circuits[log].append('DEPOLARIZE1', self.code_qubits, arg=self.sglg_err) if self.sglg_err > 0 else None
                self.circuits[log].append('TICK')

    def _preparation(self):
        for log in ["0", "1"]:
            self.circuits[log].append('R', self.qubits)
            # self.circuits[log].append('X_ERROR', self.qubits, arg=self.readout_err) if self.readout_err > 0 else None 
            # IGNORE here because first reset should not be faulty 
            if self._xbasis:
                self.circuits[log].append('H', self.code_qubits)
                self.circuits[log].append('DEPOLARIZE1', self.code_qubits, arg=self.sglg_err) if self.sglg_err > 0 else None
            self.circuits[log].append('TICK')
        self._x(["1"])

    def _first_round(self):
        rec = stim.target_rec
        for log in ["0", "1"]:
            self.circuits[log] += self._X_ent_block if self._xbasis else self._Z_ent_block
            for idx, _ in enumerate(self.link_qubits):
                self.circuits[log].append(
                    'DETECTOR', [rec(-(self.d-1)+idx)], [1+idx*2+1, 0])
            self.circuits[log].append('SHIFT_COORDS', [], (0, 1))

    def _subsequent_rounds(self):
        ent_block = self._X_ent_block if self._xbasis else self._Z_ent_block
        rounds_to_repeat = self.T - 1 if self._resets else self.T - 2
        for log in ["0", "1"]:
            self.circuits[log] += (ent_block + self._det_block) * rounds_to_repeat

    def _final_readout(self):
        rec = stim.target_rec
        for log in ["0", "1"]:
            self.circuits[log].append('H', self.code_qubits) if self._xbasis else None
            readout = 'MR' if self._resets else 'M'
            self.circuits[log].append('X_ERROR', self.code_qubits, arg=self.hard_err) if self.hard_err > 0 else None
            self.circuits[log].append(readout, self.code_qubits, arg=self.soft_err)
            self.circuits[log].append('X_ERROR', self.code_qubits, arg=self.readout_err) if self.readout_err > 0 and self._resets else None # Active reset error
            for idx, _ in enumerate(self.link_qubits):
                if not self._resets:
                    rec_list = [-self.d+idx, -self.d+idx+1, -self.d+idx-(self.d-1), -self.d+idx-2*(self.d-1)]  # [code_1, code_2, link T-1, link T-2]
                    self.circuits[log].append(
                        'DETECTOR', [rec(idx) for idx in rec_list], [1+idx*2+1, 0])
                else:
                    rec_list = [-self.d+idx, -self.d+idx+1, -self.d+idx-(self.d-1)]  # [code_1, code_2, link T-1]
                    self.circuits[log].append(
                        'DETECTOR', [rec(idx) for idx in rec_list], [1+idx*2+1, 0])
                
            self.circuits[log].append('OBSERVABLE_INCLUDE', [rec(-1)], 0)

    def _create_blocks(self):
        rec = stim.target_rec  # For readability

        # Before round depolarization due to idling
        # self._Z_ent_block.append('DEPOLARIZE1', self.code_qubits, arg=self.idle_err) if self.idle_err > 0 else None
        # self._X_ent_block.append('DEPOLARIZE1', self.code_qubits, arg=self.idle_err) if self.idle_err > 0 else None
        errs = (self.t1_err/2, self.t1_err/2, self.t2_err)
        self._Z_ent_block.append('PAULI_CHANNEL_1', self.code_qubits, arg=errs) if self.t1_err > 0 or self.t2_err > 0 else None

        # Z Entanglement block
        # L->R CXss
        LtR_indices = [t for tuple in zip(
            self.code_qubits[:-1], self.link_qubits) for t in tuple]
        self._Z_ent_block.append('CX', LtR_indices)
        self._Z_ent_block.append('DEPOLARIZE2', LtR_indices, arg=self.twog_err) if self.twog_err > 0 else None
        self._Z_ent_block.append('DEPOLARIZE1', self.code_qubits[-1], arg=self.twog_err) if self.twog_err > 0 and self.subsampling else None 
        # Extra error due to subsampling => the boundary code qubits are involved in an extra CX
        self._Z_ent_block.append('TICK')
        # R->L CXs
        RtL_indices = [t for tuple in zip(
            self.code_qubits[1:], self.link_qubits) for t in tuple]
        self._Z_ent_block.append('CX', RtL_indices)
        self._Z_ent_block.append('DEPOLARIZE2', RtL_indices, arg=self.twog_err) if self.twog_err > 0 else None
        self._Z_ent_block.append('DEPOLARIZE1', self.code_qubits[0], arg=self.twog_err) if self.twog_err > 0 and self.subsampling else None
        self._Z_ent_block.append('TICK')
        # Measure
        readout = 'MR' if self._resets else 'M'
        self._Z_ent_block.append('X_ERROR', self.link_qubits, arg=self.hard_err) if self.hard_err > 0 else None
        self._Z_ent_block.append(readout, self.link_qubits, arg=self.soft_err)
        self._Z_ent_block.append('X_ERROR', self.link_qubits, arg=self.readout_err) if self.readout_err > 0 and self._resets else None # Active reset error

        # X Entanglement block
        self._X_ent_block.append('H', self.link_qubits)
        self._X_ent_block.append('DEPOLARIZE1', self.link_qubits, arg=self.sglg_err) if self.sglg_err > 0 else None
        # L->R CXs
        LtR_indices = [t for tuple in zip(
            self.link_qubits, self.code_qubits[:-1]) for t in tuple]
        self._X_ent_block.append('CX', LtR_indices)
        self._X_ent_block.append('DEPOLARIZE2', LtR_indices, arg=self.twog_err) if self.twog_err > 0 else None
        self._X_ent_block.append('DEPOLARIZE1', self.code_qubits[-1], arg=self.twog_err) if self.twog_err > 0 and self.subsampling else None
        self._X_ent_block.append('TICK')
        # R->L CXs
        RtL_indices = [t for tuple in zip(
            self.link_qubits, self.code_qubits[1:]) for t in tuple]
        self._X_ent_block.append('CX', RtL_indices)
        self._X_ent_block.append('DEPOLARIZE2', RtL_indices, arg=self.twog_err) if self.twog_err > 0 else None
        self._X_ent_block.append('DEPOLARIZE1', self.code_qubits[0], arg=self.twog_err) if self.twog_err > 0 and self.subsampling else None
        self._X_ent_block.append('TICK')

        self._X_ent_block.append('H', self.link_qubits)
        self._X_ent_block.append('DEPOLARIZE1', self.link_qubits, arg=self.sglg_err) if self.sglg_err > 0 else None
        # Measure
        readout = 'MR' if self._resets else 'M'
        self._X_ent_block.append('X_ERROR', self.link_qubits, arg=self.hard_err) if self.hard_err > 0 else None
        self._X_ent_block.append(readout, self.link_qubits, arg=self.soft_err)
        self._X_ent_block.append('X_ERROR', self.link_qubits, arg=self.readout_err) if self.readout_err > 0 and self._resets else None

        # Detector block
        if self._resets:
            for idx, _ in enumerate(self.link_qubits):
                self._det_block.append(
                    'DETECTOR', [rec(-(self.d-1)+idx), rec(-2*(self.d-1)+idx)], [1+idx*2, 0]) # T-1 msmts
        else: # No Reset detectors
            for idx, _ in enumerate(self.link_qubits):
                self._det_block.append(
                    'DETECTOR', [rec(-(self.d-1)+idx), rec(-3*(self.d-1)+idx)], [1+idx*2+1, 0]) # T-2 msmts
        # Shift coords
        self._det_block.append('SHIFT_COORDS', [], (0, 1))
