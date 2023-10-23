# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-23

from qiskit_ibm_provider import IBMProvider
from qiskit.providers.backend import BackendV1 as Backend
from typing import Optional, Any


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
