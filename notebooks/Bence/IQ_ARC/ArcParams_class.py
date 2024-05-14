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