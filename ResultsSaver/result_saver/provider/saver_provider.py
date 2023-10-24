import gzip
import json
import os
import warnings
import logging
import datetime
import inspect
from typing import Optional, Any, Union


from qiskit.providers import JobV1, ProviderV1, JobStatus

from result_saver.job import SavedJob
from result_saver.json import ResultSaverDecoder, ResultSaverEncoder

from qiskit_ibm_provider import IBMProvider
from qiskit.providers.backend import BackendV1 as Backend


def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    warn_msg = f"Custom Warning: {message}\n"
    print(warn_msg, file=file)


# Override the default showwarning function
warnings.showwarning = custom_showwarning


def get_notebook_name():
    frame = inspect.currentframe()
    while frame:
        notebook_name = frame.f_globals.get('__vsc_ipynb_file__')
        if notebook_name:
            return os.path.basename(notebook_name).replace('.ipynb', '').replace(' ', '_')
        frame = frame.f_back  # Move to the previous frame in the call stack
    warnings.warn("Notebook name not found.")
    return 'not_defined_'  # Return a default value if the notebook name wasn't found


def update_metadata(job, backend_name, additional_dict: dict):
    '''Updates the metadata file with MANDATORY additional dicts.'''

    ROOT_DIR = find_and_create_scratch()

    notebook_name = get_notebook_name()

    now = datetime.datetime.now()
    day_str = now.strftime('%d.%m.%y')
    time_str = now.strftime('%H:%M')

    metadata_file_path = os.path.join(ROOT_DIR, f"job_metadata.json")

    # Load or create the metadata dictionary
    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, 'r') as f:
            metadata_list = json.load(f)
    else:
        metadata_list = []

    # Create new metadata entry
    metadata_entry = {
        'notebook_name': notebook_name,
        'job_id': job.job_id(),
        'backend_name': backend_name,
        'job_name': job.name(),
        'job_metadata': job.metadata,
        "tags": job.tags(),
        "creation_date": str(job.creation_date()),
        'day_str': day_str,
        'time_str': time_str,
    }

    if additional_dict:
        metadata_entry.update(additional_dict)

    # Append the new metadata entry to the list
    metadata_list.append(metadata_entry)

    # Save updated metadata
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    pass


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
                warnings.warn("Soft-Info folder not found.")
                break
    os.chdir(original_path)  # Navigate back to original directory
    return scratch_path


class SaverProvider(IBMProvider):
    ROOT_DIR = find_and_create_scratch()
    DEFAULT_SAVE_LOCATION = f"{ROOT_DIR}/jobs"
    FORMAT = "{date_str}-{job_id}.json.gz"

    def __init__(self, save_location: Optional[str] = None) -> None:
        super().__init__()
        self.save_location = save_location or self.DEFAULT_SAVE_LOCATION

    def get_backend(self,
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

    def new_run(self, metadata: dict, *args, **kwargs):
        warnings.warn("updating metadata")
        # Call the original run method
        job = self.original_run(*args, **kwargs)
        # Update metadata
        update_metadata(job, self.name, metadata)
        return job

    def retrieve_job(self, job_id: str) -> JobV1:
        """Return a single job.

        Args:
            job_id: The ID of the job to retrieve.

        Returns:
            The job with the given id.
        """
        filename = self.__job_local_filename(job_id)
        if filename is None:
            warnings.warn(
                f"Job ID {job_id} not found in {self.save_location}. Retrieving it from the IBMQ provider...")
            ibm_prov = IBMProvider()
            ibm_job = ibm_prov.retrieve_job(job_id)
            if ibm_job.status() in [JobStatus.RUNNING, JobStatus.QUEUED]:
                warnings.warn(f"Job ID {job_id} is still running. Aborting...")
                return None
            if ibm_job.status() == JobStatus.DONE:
                self.save_job(ibm_job)
                return ibm_job
            else:
                warnings.warn(
                    f"Job ID {job_id} is in an unknown state. Aborting...")
                return None

        try:
            with gzip.GzipFile(filename, "r") as f:
                job_str = str(f.read(), "utf8")
            job = json.loads(job_str, cls=ResultSaverDecoder)
            return job
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve job for job id {job_id}.", e)

    def __save_location(self):
        return os.path.expanduser(self.save_location)

    def __job_saved_name(self, job_id: str) -> str:
        date_str = datetime.datetime.now().strftime("%Y.%m.%d-%Hh%M")
        return self.FORMAT.format(date_str=date_str, job_id=job_id)

    def __job_local_filename(self, job_id: str) -> Union[str, None]:
        save_location = self.__save_location()
        for filename in os.listdir(save_location):
            if job_id in filename:
                return os.path.join(save_location, filename)
        return None

    def __create_dir_if_doesnt_exist(self):
        if not os.path.exists(self.__save_location()):
            os.makedirs(self.__save_location())

    def save_job(self, job: JobV1, overwrite: bool = False) -> str:
        if self.__job_local_filename(job.job_id()) and not overwrite:
            warnings.warn(
                f"Job ID {job.job_id()} already saved and overwrite=False. Skipping...")
            return None

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
