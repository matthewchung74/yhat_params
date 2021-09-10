import io
import json
import os
import time
from typing import Optional
import uuid
from enum import Enum
from functools import wraps
from io import BytesIO
from pathlib import Path
import PIL

import boto3
import cv2
import numpy as np
import requests
from PIL import Image

AWS_REQUEST_BUCKET = os.getenv("AWS_REQUEST_BUCKET")


class FieldType(str, Enum):
    Text = "Text"
    # OpenCV = "OpenCV"
    PIL = "PIL"


root_dir = Path("/tmp")
nl = "\n"
default_image = "http://images.cocodataset.org/val2017/000000439715.jpg"


def get_s3_resource() -> boto3.session.Session.resource:
    if os.getenv("AWS_ACCESS_KEY") == None:
        return boto3.resource("s3")
    else:
        return boto3.resource(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            region_name=os.getenv("AWS_REGION_NAME"),
        )


def write_image_to_s3(image, s3_uri: str):
    bucket = Path(s3_uri).parts[1]
    key = "/".join(list(Path(s3_uri).parts[2:]))
    s3_bucket = get_s3_resource().Bucket(bucket)
    object = s3_bucket.Object(key)
    file_stream = BytesIO()
    if "PIL" not in str(type(image)):
        image = Image.fromarray(image)

    try:
        image.save(file_stream, format="jpeg")
        object.put(Body=file_stream.getvalue())
    except Exception as e:
        print(e)


def read_image_from_s3(s3_uri: str, field_type: FieldType):
    bucket = Path(s3_uri).parts[1]
    key = "/".join(list(Path(s3_uri).parts[2:]))
    s3_bucket = get_s3_resource().Bucket(bucket)
    object = s3_bucket.Object(key)
    response = object.get()
    file_stream = response["Body"]
    im = Image.open(file_stream)
    if field_type == FieldType.PIL:
        return im
    else:
        # return np.array(im)
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def delete_images_from_s3(bucket: Optional[str], object_prefix: str = "unit_test/"):
    if bucket == None:
        raise Exception(f"bucket:{bucket} not found")

    s3_bucket = get_s3_resource().Bucket(bucket)
    objects = s3_bucket.objects.all()
    for object in objects:
        object.delete()


def grab_image(field_type: FieldType, url: str = default_image):
    url_path = Path(url)
    image_name = url_path.name
    protocol = url_path.parts[0]
    if protocol == "http:" or protocol == "https:":
        data = requests.get(url, allow_redirects=True).content
        im = Image.open(io.BytesIO(data))
        if field_type == FieldType.PIL:
            return im
        # elif field_type == FieldType.OpenCV:
        #     return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        # return np.array(img)
        else:
            raise Exception(f"Unknown type found {field_type}")
    elif protocol == "s3:":
        bucket = url_path.parts[1]
        key = "/".join(list(url_path.parts[2:]))
        return read_image_from_s3(s3_uri=url, field_type=field_type)


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in FieldType.values():
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def in_colab():
    try:
        import google.colab  # type: ignore

        return True
    except:
        return False


def clean_json():
    if (root_dir / "input.json").exists():
        (root_dir / "input.json").unlink()
    if (root_dir / "output.json").exists():
        (root_dir / "output.json").unlink()
    if (root_dir / "result.json").exists():
        (root_dir / "result.json").unlink()


def inference_predict(input: dict, output: dict):
    clean_json()

    expect_json = json.dumps(input, cls=EnumEncoder)
    with open(root_dir / "input.json", "w") as f:
        f.write(expect_json)

    output_json = json.dumps(output, cls=EnumEncoder)
    if not Path(root_dir / "output.json").exists() or in_colab():
        with open(root_dir / "output.json", "w") as f:
            f.write(output_json)

    def inference_predict_decorator(func):
        @wraps(func)
        def inference_predict_wrapper(*args, **kwargs):
            begin = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            return (result, end - begin)

        return inference_predict_wrapper

    return inference_predict_decorator


def convert_input_thumb(
    s3_uri_source: str,
    param_value: PIL,
    field_type: FieldType,
    bucket_name: str,
    object_prefix: str,
):
    url_path = Path(s3_uri_source)
    image_name = url_path.stem
    protocol = url_path.parts[0]
    if protocol == "s3:":
        # if field_type == FieldType.OpenCV:
        #     param_value = Image.fromarray(param_value)

        param_value = param_value.convert("RGB")
        param_value.thumbnail((300, 300))

        file_key = f"{object_prefix}/{image_name}-thumb.jpg"
        bucket = bucket_name if bucket_name != None else AWS_REQUEST_BUCKET
        s3_uri_dest: str = f"s3://{bucket}/{file_key}"
        write_image_to_s3(image=param_value, s3_uri=s3_uri_dest)


def convert_input_params(params: dict, bucket_name: str, object_prefix: str):
    check_input_params(params=params, assets_downloaded=False)
    with open(root_dir / "input.json") as f:
        input_format = json.load(f)

    for param_key in params:
        param_value = params[param_key]
        if (
            input_format[param_key]
            == FieldType.PIL
            # or input_format[param_key] == FieldType.OpenCV
        ):
            params[param_key] = grab_image(
                field_type=input_format[param_key], url=param_value
            )

            # if input_format[param_key] == FieldType.OpenCV:
            #     params[param_key] = cv2.cvtColor(params[param_key], cv2.COLOR_BGR2RGB)

            convert_input_thumb(
                s3_uri_source=param_value,
                param_value=params[param_key],
                field_type=input_format[param_key],
                bucket_name=bucket_name,
                object_prefix=object_prefix,
            )
    check_input_params(params=params, assets_downloaded=True)
    return params


def check_input_params(params: dict, assets_downloaded: bool = True):
    if not Path(root_dir / "input.json").exists():
        raise Exception(f"missing {root_dir/'input.json'}")
    else:
        with open(root_dir / "input.json") as f:
            input_format = json.load(f)

    if type(params) is not dict:
        raise Exception("params needs to be a dictionary")

    # check for invalid input keys
    for param_key in params:
        if param_key not in input_format.keys():
            message = f"""
                @inference_predict(input=input, output=output) input parameter needs to have a key that is  
                one of the following: 
                    {nl.join(list(input_format.keys()))}
                invalid key found:
                    {param_key}
                """
            raise Exception(message)

        # check for invalid input values for text
        param_value = params[param_key]
        if input_format[param_key] == FieldType.Text:
            if not type(param_value) == type("str"):
                message = f"""
                    @inference_predict(input=input, output=output) input parameter  
                        {param_key}
                    needs to be of type:
                        FieldType.Text {type('str')}
                        Found ({type(param_value)}) 
                    """
                raise Exception(message)

        # check for invalid input values for PIL
        if input_format[param_key] == FieldType.PIL:
            if assets_downloaded and "PIL." not in str(type(param_value)):
                message = f"""
                    @inference_predict(input=input, output=output) input parameter  
                        {param_key}
                    needs to be of type:
                        FieldType.PIL {type(Image.new('RGB', (1, 1), color = 'white'))}
                        Found ({type(param_value)}) 
                    """
                raise Exception(message)

        # # check for invalid input values for cv2
        # if input_format[param_key] == FieldType.OpenCV:
        #     if assets_downloaded and type(param_value) != type(
        #         np.zeros(shape=(1, 1, 3), dtype=np.uint8)
        #     ):
        #         message = f"""
        #             @inference_predict(input=input, output=output) input parameter
        #                 {param_key}
        #             needs to be of type:
        #                 FieldType.OpenCV {type(np.zeros(shape=(1,1,3), dtype=np.uint8))}
        #                 Found ({type(param_value)})
        #             """
        #         raise Exception(message)


def convert_output_params(result: dict, object_prefix: str, bucket_name: str) -> dict:
    check_output_params(result)
    with open(root_dir / "output.json") as f:
        output_format = json.load(f)

    for param_key in result:
        param_value = result[param_key]
        if (
            output_format[param_key]
            == FieldType.PIL
            # or output_format[param_key] == FieldType.OpenCV
        ):
            file_key = f"{object_prefix}/{uuid.uuid1()}"
            bucket = bucket_name if bucket_name != None else AWS_REQUEST_BUCKET
            s3_uri = f"s3://{bucket}/{file_key}.jpg"
            write_image_to_s3(
                image=param_value,
                s3_uri=s3_uri,
            )

            convert_input_thumb(
                s3_uri_source=s3_uri,
                param_value=param_value,
                field_type=output_format[param_key],
                bucket_name=bucket_name,
                object_prefix=object_prefix,
            )
            result[param_key] = s3_uri
    return result


def check_output_params(result: dict):
    if not Path(root_dir / "output.json").exists():
        raise Exception("missing output.json")
    else:
        with open(root_dir / "output.json") as f:
            output_format = json.load(f)

    # check for invalid result keys
    for param_key in result:
        if param_key not in output_format.keys():
            message = f"""
                @inference_predict(input=input, output=output) output parameter needs to have a key that is  
                one of the following: 
                    {nl.join(list(output_format.keys()))}
                invalid key found:
                    {param_key}
                """
            raise Exception(message)

        # check for invalid result text values
        param_value = result[param_key]
        if output_format[param_key] == FieldType.Text:
            if not type(param_value) == type("str"):
                message = f"""
                    @inference_predict(input=input, output=output) output parameter  
                        {param_key}
                    needs to be of type:
                        FieldType.Text {type('str')}
                        Found ({type(param_value)}) 
                    """
                raise Exception(message)

        # check for invalid result PIL values
        if output_format[param_key] == FieldType.PIL:
            if "PIL." not in str(type(param_value)):
                message = f"""
                    @inference_predict(input=input, output=output) output parameter  
                        {param_key}
                    needs to be of type:
                        FieldType.PIL {type(Image.new('RGB', (1, 1), color = 'white'))}
                        Found ({type(param_value)}) 
                    """
                raise Exception(message)

        # # check for invalid result OpenCV values
        # if output_format[param_key] == FieldType.OpenCV:
        #     if type(param_value) != type(np.zeros(shape=(1, 1, 3), dtype=np.uint8)):
        #         message = f"""
        #             @inference_predict(input=input, output=output) output parameter
        #                 {param_key}
        #             needs to be of type:
        #                 FieldType.OpenCV {type(np.zeros(shape=(1,1,3), dtype=np.uint8))}
        #                 Found ({type(param_value)})
        #             """
        #         raise Exception(message)


def inference_test(predict_func, params: dict):

    check_input_params(params=params)

    result, duration = predict_func(params)

    check_output_params(result=result)

    with open(root_dir / "result.json", "w") as f:
        if Path(root_dir / "result.jpg").exists():
            Path(root_dir / "result.jpg").unlink()

        for index, key in enumerate(result):
            if "PIL." in str(type(result[key])):
                result[key].save(f"{key}{index}.jpg")
                result[key] = f"./{key}{index}.jpg"
            if type(result[key]) == type(np.zeros(shape=(1, 1, 3), dtype=np.uint8)):
                cv2.imwrite(f"{key}{index}.jpg", result[key])
                result[key] = f"./{key}{index}.jpg"

        json.dump(result, f)
        print(f"Wrote results to result.json duration: {round(duration,6)} seconds")
        print(f"Please take a look and verify the results")
        print(f"{json.dumps(result, indent=4)}")
