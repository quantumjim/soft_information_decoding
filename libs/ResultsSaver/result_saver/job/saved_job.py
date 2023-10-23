from typing import Optional, List
from qiskit.providers import JobV1
from qiskit.providers.backend import Backend
from qiskit.result import Result


class SavedJob(JobV1):
    def __init__(
        self,
        backend: Backend | None,
        job_id: str,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._tags = tags
        self._name = name
        self._result = None
        self._creation_date = kwargs.pop("creation_date", None)
        super().__init__(backend, job_id, **kwargs)

    def set_result(self, result: Result):
        self._result = result

    def creation_date(self):
        return self._creation_date

    def result(self, *args, **kwargs):
        return self._result

    @classmethod
    def from_job(cls, job: JobV1) -> "SavedJob":
        if hasattr(job, "tags"):
            tags = job.tags()
        else:
            tags = None
        if hasattr(job, "name"):
            name = job.name()
        else:
            name = None
        if hasattr(job, "creation_date"):
            creation_date = job.creation_date()
        else:
            creation_date = None
        if hasattr(job,)
        new_job: "SavedJob"
        new_job = cls(
            job.backend(),
            job.job_id(),
            tags=tags,
            name=name,
            creation_date=creation_date,
            time_per_step=
            **job.metadata,
        )
        new_job.set_result(job.result())
        return new_job

    def submit(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support submitting jobs."
        )

    def status(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support checking the status."
        )

    def tags(self):
        return self._tags

    def name(self):
        return self._name
