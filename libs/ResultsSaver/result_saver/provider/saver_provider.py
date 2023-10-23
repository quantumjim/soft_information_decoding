import gzip
import json
import os
from typing import Optional

from qiskit.providers import JobV1, ProviderV1

from result_saver.job import SavedJob
from result_saver.json import ResultSaverDecoder, ResultSaverEncoder


class SaverProvider(ProviderV1):
    DEFAULT_SAVE_LOCATION = "~/.saved_jobs"
    FORMAT = "job_{job_id}.json.gz"

    def __init__(self, save_location: Optional[str] = None) -> None:
        super().__init__()

        self.save_location = save_location or self.DEFAULT_SAVE_LOCATION

    def backends(self, name=None, **kwargs):
        return []

    def retrieve_job(self, job_id: str) -> JobV1:
        """Return a single job.

        Args:
            job_id: The ID of the job to retrieve.

        Returns:
            The job with the given id.
        """
        if not self.__job_is_saved(job_id):
            raise RuntimeError(f"Job ID {job_id} not found in {self.save_location}.")
            # ibm_job = ibm_prov.retrieve_job(job_id)
            # self.save_job(ibm_job)

        try:
            filename = os.path.join(
                self.__save_location(), self.__job_saved_name(job_id)
            )
            with gzip.GzipFile(filename, "r") as f:
                job_str = str(f.read(), "utf8")
            job = json.loads(job_str, cls=ResultSaverDecoder)
            return job
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve job for job id {job_id}.", e)

    def __save_location(self):
        return os.path.expanduser(self.save_location)

    def __job_saved_name(self, job_id: str) -> str:
        return self.FORMAT.format(job_id=job_id)

    def __job_is_saved(self, job_id: str) -> bool:
        return os.path.exists(
            os.path.join(self.__save_location(), self.__job_saved_name(job_id))
        )

    def __create_dir_if_doesnt_exist(self):
        if not os.path.exists(self.__save_location()):
            os.makedirs(self.__save_location())

    def save_job(self, job: JobV1, overwrite: bool = False) -> str:
        if self.__job_is_saved(job) and not overwrite:
            raise RuntimeError("Job is already saved and overwrite=False.")
        job = SavedJob.from_job(job)
        try:
            self.__create_dir_if_doesnt_exist()
            filename = os.path.join(
                self.__save_location(), self.__job_saved_name(job.job_id())
            )
            job_str = json.dumps(job, cls=ResultSaverEncoder)
            with gzip.GzipFile(filename, "w") as f:
                f.write(bytes(job_str, "utf8"))
            return filename
        except Exception as e:
            raise RuntimeError(f"Failed to save job {job.job_id()}", e)
