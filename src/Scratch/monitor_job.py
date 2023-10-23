# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-23

import sys
import json
from time import sleep

from qiskit.providers.jobstatus import JobStatus

from .core import get_job_data

def monitor_and_save_jon(job_id, provider, additional_dict=None):
    job = provider.retrieve_job(job_id)

    while job.status() in [JobStatus.RUNNING, JobStatus.QUEUED]:
        sleep(120)

    get_job_data(job_id, provider, additional_dict=additional_dict)


if __name__ == '__main__':
    job_id = sys.argv[1]
    provider = sys.argv[2]
    additional_dict_str = sys.argv[3]
    additional_dict = json.loads(additional_dict_str)
    monitor_and_save_jon(job_id, provider, additional_dict=additional_dict)