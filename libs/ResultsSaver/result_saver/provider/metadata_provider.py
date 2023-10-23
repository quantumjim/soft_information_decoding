# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-23

from qiskit_ibm_provider import IBMProvider
from qiskit.providers.backend import BackendV1 as Backend
from typing import Optional, Any, Union

def metadata_helper(n_shots: Union[int, float], meas_level: Union[1, 2], *args, **kwargs):
    """
    A helper function to collect and organize metadata.

    Parameters:
    n_shots (Union[int, float]): The number of shots for the experiment.
    meas_level (Union[1, 2]): The measurement level, either 1 or 2.
    *args: Additional arguments.
    **kwargs: Additional keyword arguments.

    Returns:
    Dict[str, Any]: A dictionary containing the collected metadata.
    """

    # Organize the metadata into a dictionary
    metadata = {
        'n_shots': n_shots,
        'meas_level': meas_level
    }

    # If there are additional arguments or keyword arguments, include them in the metadata
    if args:
        metadata['additional_args'] = args
    if kwargs:
        metadata.update(kwargs)

    return metadata

class MetadataProvider(IBMProvider):

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
