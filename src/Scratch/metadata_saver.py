# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-20

from qiskit_ibm_provider import IBMProvider

class MyCustomProvider(IBMProvider):
    
    def get_backend(self, name=None, **kwargs):
        backend = super().get_backend(name, **kwargs)
        self.patch_backend(backend)
        return backend

    def patch_backend(self, backend):
        if not hasattr(backend, 'original_run'):  # Avoid patching multiple times
            backend.original_run = backend.run  # Store the original run method
            backend.run = self.new_run.__get__(backend)  # Replace run with new_run

    def new_run(self, *args, **kwargs):
        print("Provider: Running additional functions before backend.run")
        job = self.original_run(*args, **kwargs)  # Call the original run method
        print("Provider: Running additional functions after backend.run")
        return job