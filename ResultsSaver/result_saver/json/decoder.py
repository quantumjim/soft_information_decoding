"""Result Saver Decoder."""

from datetime import datetime

from qiskit.result import Result
from qiskit_experiments.framework.json import ExperimentDecoder

from result_saver.job import SavedJob


class ResultSaverDecoder(ExperimentDecoder):
    def __decode_datetime(self, datetime_str: str):
        return datetime.fromisoformat(datetime_str)

    def __decode_ibmjob(self, job_dict: dict) -> SavedJob:
        # "job_id": obj.job_id(),
        # "result": obj.result(),
        # "metadata": obj.metadata,
        # "header": obj.header(),
        # "tags": obj.tags(),
        job = SavedJob(
            backend=job_dict["backend"],
            api_client=None,
            job_id=job_dict["job_id"],
            tags=job_dict["tags"],
            name=job_dict["name"],
            creation_date=job_dict["creation_date"],
            **job_dict["metadata"],
        )
        job.set_result(job_dict["result"])
        return job

    def __decode_result(self, result_dict: dict) -> Result:
        return Result.from_dict(result_dict)

    def object_hook(self, obj: dict):
        if "__type__" in obj:
            obj_type = obj["__type__"]
            obj_value = obj["__value__"]
            if obj_type == "datetime":
                return self.__decode_datetime(obj_value)
            if obj_type == "SavedJob":
                return self.__decode_ibmjob(obj_value)
            if obj_type == "qiskit.result.Result":
                return self.__decode_result(obj_value)
        return super().object_hook(obj)
