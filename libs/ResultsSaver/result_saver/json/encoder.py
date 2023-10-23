"""Result Saver Encoder."""

from datetime import datetime
from typing import Any

from qiskit.result import Result
from qiskit_experiments.framework.json import ExperimentEncoder
from qiskit_ibm_provider.job import IBMJob

from result_saver.job import SavedJob


class ResultSaverEncoder(ExperimentEncoder):
    def __encode_datetime(self, obj: datetime):
        return {"__type__": "datetime", "__value__": obj.isoformat()}

    def __encode_job(self, obj: SavedJob):
        return {
            "__type__": "SavedJob",
            "__value__": {
                "job_id": obj.job_id(),
                "result": obj.result(),
                "metadata": obj.metadata,
                "tags": obj.tags(),
                "name": obj.name(),
                "creation_date": obj.creation_date(),
                "backend": str(obj.backend()),
            },
        }

    def __encode_result(self, obj: Result):
        return {
            "__type__": "qiskit.result.Result",
            "__value__": obj.to_dict(),
        }

    def default(self, obj: Any) -> Any:
        if isinstance(obj, SavedJob):
            return self.__encode_job(obj)
        if isinstance(obj, datetime):
            return self.__encode_datetime(obj)
        if isinstance(obj, Result):
            return self.__encode_result(obj)
        return super().default(obj)
