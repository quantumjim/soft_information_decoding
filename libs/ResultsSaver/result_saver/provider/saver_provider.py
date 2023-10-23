import gzip
import json
import os
import warnings
from typing import Optional, Any


from qiskit.providers import JobV1, ProviderV1

from result_saver.job import SavedJob
from result_saver.json import ResultSaverDecoder, ResultSaverEncoder

from qiskit_ibm_provider import IBMProvider
from qiskit.providers.backend import BackendV1 as Backend

def find_and_create_scratch():
    original_path = os.getcwd()
    scratch_path = None
    while True:
        current_folder = os.path.basename(os.getcwd())
        if current_folder == 'Soft-Info':
            scratch_path = os.path.join(os.getcwd(), '.Scratch')
            if not os.path.exists('.Scratch'):
                os.mkdir('.Scratch')
            break
        else:
            os.chdir('..')
            if os.getcwd() == '/':  # Stop if we reach the root directory
                print("Soft-Info folder not found.")
                break
    os.chdir(original_path)  # Navigate back to original directory
    return scratch_path


class SaverProvider(IBMProvider):
    DEFAULT_SAVE_LOCATION = f"{find_and_create_scratch()}/jobs"
    FORMAT = "job_{job_id}.json.gz"

    def __init__(self, save_location: Optional[str] = None) -> None:
        super().__init__()
        self.save_location = save_location or self.DEFAULT_SAVE_LOCATION
    
    def get_backend( self,
        name: str = None,
        instance: Optional[str] = None,
        **kwargs: Any,
    ) -> Backend:
        """Return a monkey patched backend."""
        backend = super().get_backend(name, **kwargs)
        self.patch_backend(backend)
        return backend

    def patch_backend(self, backend):
        if not hasattr(backend, 'original_run'):  # Avoid patching multiple times
            backend.original_run = backend.run  # Store the original run method
            backend.run = self.new_run.__get__(
                backend)  # Replace run with new_run

    def new_run(self, *args, **kwargs):
        print("Provider: Running additional functions before backend.run")
        # Call the original run method
        job = self.original_run(*args, **kwargs)
        print("Provider: Running additional functions after backend.run")
        return job


    def retrieve_job(self, job_id: str) -> JobV1:
        """Return a single job.

        Args:
            job_id: The ID of the job to retrieve.

        Returns:
            The job with the given id.
        """
        if not self.__job_is_saved(job_id):
            warnings.warn(f"Job ID {job_id} not found in {self.save_location}. Retrieving it from the IBMQ provider...")         
            ibm_prov = IBMProvider()
            ibm_job = ibm_prov.retrieve_job(job_id)
            self.save_job(ibm_job)

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
        if self.__job_is_saved(job.job_id()) and not overwrite:
            warnings.warn(f"Job ID {job.job_id()} already saved and overwrite=False. Skipping...")

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
