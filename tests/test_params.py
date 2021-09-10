from sys import prefix
from typing import Optional

# from botocore.retries import bucket
from yhat_params.yhat_tools import (
    inference_predict,
    inference_test,
    grab_image,
    read_image_from_s3,
    write_image_to_s3,
    delete_images_from_s3,
    convert_input_params,
    convert_output_params,
)
from yhat_params.yhat_tools import FieldType

# import cv2
from PIL import Image
import boto3
import os

# import numpy as np

object_prefix = "unit_test"

aws_bucket: Optional[str] = os.getenv("AWS_REQUEST_BUCKET")
aws_access_key_id = (os.getenv("AWS_ACCESS_KEY"),)
aws_secret_access_key = (os.getenv("AWS_SECRET_KEY"),)
aws_region_name = os.getenv("AWS_REGION_NAME")

if (
    aws_access_key_id == None
    or aws_secret_access_key == None
    or aws_region_name == None
    or aws_bucket == None
):
    raise Exception("Missing aws env vars")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region_name,
)


def upload_to_s3(file_key: str) -> str:
    delete_images_from_s3(bucket=aws_bucket, object_prefix=object_prefix)
    pil_image = grab_image(field_type=FieldType.PIL)
    s3_uri = f"s3://{aws_bucket}/{file_key}"
    write_image_to_s3(image=pil_image, s3_uri=s3_uri)
    return f"s3://{aws_bucket}/{file_key}"


def test_read_write_s3():
    # cv_image1 = cv2.imread("tests/red.jpg")
    # s3_uri = f"s3://{aws_bucket}/unit_test/cv_image.jpg"
    # write_image_to_s3(image=cv_image1, s3_uri=s3_uri)
    # cv_image2 = read_image_from_s3(s3_uri=s3_uri, field_type=FieldType.OpenCV)
    # cv2.imwrite("/tmp/cv_image_test1.jpg", cv_image2)

    pil_image1 = Image.open("tests/red.jpg")
    s3_uri = f"s3://{aws_bucket}/unit_test/pil_image.jpg"
    write_image_to_s3(image=pil_image1, s3_uri=s3_uri)
    pil_image2 = read_image_from_s3(s3_uri=s3_uri, field_type=FieldType.PIL)
    pil_image2.save("/tmp/pil_image_test2.jpg")


def test_str_str():
    input = {"text input": FieldType.Text}
    output = {"result": FieldType.Text}

    @inference_predict(input=input, output=output)
    def predict(params):
        return {"result": "positive and 100% accurate"}

    inference_test(predict_func=predict, params={"text input": "loved that movie"})


def test_pil_pil():
    input = {"image input": FieldType.PIL}
    output = {"result image": FieldType.PIL, "result text": FieldType.Text}

    @inference_predict(input=input, output=output)
    def predict(params):
        return {"result image": grab_image(FieldType.PIL), "result text": "abc"}

    params = {"image input": grab_image(FieldType.PIL)}
    inference_test(predict_func=predict, params=params)


# def test_cv2_cv2():
#     input = {"image input": FieldType.OpenCV}
#     output = {"result image": FieldType.OpenCV, "result text": FieldType.Text}

#     @inference_predict(input=input, output=output)
#     def predict(params):
#         return {"result image": grab_image(FieldType.OpenCV), "result text": "abc"}

#     params = {"image input": grab_image(FieldType.OpenCV)}
#     inference_test(predict_func=predict, params=params)


def test_convert_input_params():
    input = {
        "text input": FieldType.Text,
        "pil input": FieldType.PIL,
    }

    # this is here to create json files
    @inference_predict(input=input, output={})
    def predict(params):
        return {"result image": grab_image(FieldType.OpenCV), "result text": "abc"}

    file_key = f"{object_prefix}/myimg.jpg"
    s3_uri = upload_to_s3(file_key=file_key)

    # make fake input params
    new_params = {"text input": "blah", "pil input": s3_uri}

    converted_params = convert_input_params(
        params=new_params, bucket_name=aws_bucket, object_prefix=object_prefix
    )
    assert "PIL" in str(converted_params["pil input"])


# def test_convert_input_params():
#     input = {
#         "text input": FieldType.Text,
#         "cv input": FieldType.OpenCV,
#         "pil input": FieldType.PIL,
#     }

#     # this is here to create json files
#     @inference_predict(input=input, output={})
#     def predict(params):
#         return {"result image": grab_image(FieldType.OpenCV), "result text": "abc"}

#     file_key = f"{object_prefix}/myimg.jpg"
#     s3_uri = upload_to_s3(file_key=file_key)

#     # make fake variables
#     new_params = {"text input": "blah", "cv input": s3_uri, "pil input": s3_uri}

#     converted_params = convert_input_params(
#         params=new_params, bucket_name=aws_bucket, object_prefix=object_prefix
#     )
#     assert type(converted_params["cv input"]) == type(np.zeros(shape=(1)))
#     assert "PIL" in str(converted_params["pil input"])


def test_convert_output_params():

    output = {
        "text output": FieldType.Text,
        "pil output": FieldType.PIL,
    }
    # this is here to create json files
    @inference_predict(input={}, output=output)
    def predict(params):
        return {"result image": grab_image(FieldType.PIL), "result text": "abc"}

    result = {
        "text output": "blah",
        "pil output": grab_image(FieldType.PIL),
    }

    converted_params = convert_output_params(
        bucket_name=aws_bucket, result=result, object_prefix=prefix
    )
    assert type(result["text output"]) == type("")
    # assert type(result["cv output"]) == type("")
    assert type(result["pil output"]) == type("")


# def test_convert_output_params():

#     output = {
#         "text output": FieldType.Text,
#         "cv output": FieldType.OpenCV,
#         "pil output": FieldType.PIL,
#     }
#     # this is here to create json files
#     @inference_predict(input={}, output=output)
#     def predict(params):
#         return {"result image": grab_image(FieldType.OpenCV), "result text": "abc"}

#     result = {
#         "text output": "blah",
#         "cv output": grab_image(FieldType.OpenCV),
#         "pil output": grab_image(FieldType.PIL),
#     }

#     converted_params = convert_output_params(
#         bucket_name=aws_bucket, result=result, object_prefix=prefix
#     )
#     assert type(result["text output"]) == type("")
#     assert type(result["cv output"]) == type("")
#     assert type(result["pil output"]) == type("")
