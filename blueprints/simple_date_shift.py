"""
Drop a field from the input record, or all the fields matching a blob.
"""
from gretel_client.transformers.transformers import DateShiftConfig
from gretel_client.transformers import DataPath, DataRestorePipeline, DataTransformPipeline, FieldRef

xf_date = DateShiftConfig(secret='2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F7F036D6F04FC6A94',
                          lower_range_days=-10, upper_range_days=25,
                          tweak=FieldRef('user_id'))

data_paths = [DataPath(input='birthday', xforms=xf_date),
              DataPath(input='*')]

pipe = DataTransformPipeline(data_paths)
restore_pipe = DataRestorePipeline(data_paths)

records = [
    {
        "user_id": "michaelj@dabulls.com",
        "birthday": "02/17/1963"
    },
    {
        "user_id": "michaelj@twinpinesmall.com",
        "birthday": "06/09/1961"
    },
    {
        "user_id": "michaelj@titostacos.com",
        "birthday": "08/29/1958"
    },
]

print(f"Original records: {records}\n")

out = [pipe.transform_record(rec) for rec in records]

assert out == [
    {
        "user_id": "michaelj@dabulls.com",
        "birthday": "1963-02-13"
    },
    {
        "user_id": "michaelj@twinpinesmall.com",
        "birthday": "1961-06-05"
    },
    {
        "user_id": "michaelj@titostacos.com",
        "birthday": "1958-08-25"
    },
]

print(f"Transformed output: {out}\n")

restored = [restore_pipe.transform_record(rec) for rec in out]

# Please note the format!
assert restored == [
    {
        "user_id": "michaelj@dabulls.com",
        "birthday": "1963-02-17"
    },
    {
        "user_id": "michaelj@twinpinesmall.com",
        "birthday": "1961-06-09"
    },
    {
        "user_id": "michaelj@titostacos.com",
        "birthday": "1958-08-29"
    },
]

print(f"Restored records: {restored}\n")
