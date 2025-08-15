"""
Bedrock Batches API Handler
"""

from typing import Any, Coroutine, Optional, Union, cast

import httpx

# from litellm.llms.bedrock.bedrock import AsyncBedrock, Bedrock
from litellm.types.llms.openai import (
    Batch, 
    CancelBatchRequest, 
    CreateBatchRequest, 
    RetrieveBatchRequest
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)

from litellm.types.utils import LiteLLMBatch
from ..base_aws_llm import BaseAWSLLM
from .transformation import transform_openai_create_batch_to_bedrock_job_request
from litellm.types.llms.bedrock import CreateModelInvocationJobRequest
import litellm

class BedrockBatchesAPI(BaseAWSLLM):
    """
    Bedrock methods to support for batches
    - create_batch()
    - retrieve_batch()
    - cancel_batch()
    - list_batch()
    """

    def __init__(self) -> None:
        super().__init__()
    
    async def acreate_batch(
        self,
        create_batch_data: CreateBatchRequest,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        api_key: Optional[str] = None,
    ) -> LiteLLMBatch:
        response = await client.batches.create(**create_batch_data)
        return LiteLLMBatch(**response.model_dump())
    
    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
        organization: Optional[str],
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
    ) -> LiteLLMBatch:
        # Build boto3 client via existing BaseAWSLLM credential helpers
        from litellm.llms.bedrock.common_utils import init_bedrock_client

        # Derive input/output S3 URIs from the input_file_id returned by files.create
        file_id = create_batch_data.get("input_file_id")
        # The file_id is expected to be of the form f"s3://{bucket}/path/model_id/" or f"s3://{bucket}/{model_id}/"
        # We need to extract both the input S3 URI and the model_id
        # Example: file_id = "s3://my-bucket/path/anthropic.claude-3-sonnet-20240229-v1:0/"
        if not isinstance(file_id, str) or not file_id.startswith("s3://"):
            raise ValueError("input_file_id must be an s3:// URI for Bedrock batch jobs")
        
        # Remove the "s3://" prefix and parse the path
        s3_path = file_id[len("s3://") :].rstrip("/")
        path_parts = s3_path.split("/")
        
        if len(path_parts) < 1:
            raise ValueError("Invalid S3 URI format")
        
        bucket = path_parts[0]
        
        # Extract model_id - it should be the last path component (folder name)
        if len(path_parts) >= 2:
            model_id = path_parts[-1]  # Last part is the model_id
            # Input S3 URI is the full path up to the model_id folder
            input_s3_uri = file_id if file_id.endswith("/") else file_id + "/"
        else:
            # Only bucket, no model_id
            model_id = ""
            input_s3_uri = f"s3://{bucket}/"
            
        if not isinstance(input_s3_uri, str) or not input_s3_uri.startswith("s3://"):
            raise ValueError("input_file_id must be an s3:// URI for Bedrock batch jobs")

        # output path: same path as input but without the filename (ensure trailing slash)
        # s3://bucket/path/to/file.jsonl -> s3://bucket/path/to/
        try:
            without_scheme = input_s3_uri[len("s3://"):]
            bucket_and_key = without_scheme.split("/", 1)
            if len(bucket_and_key) == 1:
                # no key, just bucket; place outputs at bucket root
                output_s3_uri = f"s3://{bucket_and_key[0]}/"
            else:
                bucket, key = bucket_and_key[0], bucket_and_key[1]
                prefix = key.rsplit("/", 1)[0] if "/" in key else ""
                output_s3_uri = f"s3://{bucket}/{prefix}/"
        except Exception:
            # Fallback to bucket root if parsing fails
            output_s3_uri = input_s3_uri.split("/", 3)[0] + "//" + input_s3_uri.split("/", 3)[2] + "/"

        # Optional role from config
        s3_params = getattr(litellm, "s3_callback_params", {}) or {}
        role_arn = s3_params.get("s3_aws_role_name")
        if isinstance(role_arn, str) and not role_arn.startswith("arn:"):
            role_arn = None

        # Ensure the model is set in the create_batch_data if we extracted it from file_id
        if model_id and not create_batch_data.get("model"):
            create_batch_data = dict(create_batch_data)  # Make a copy
            create_batch_data["model"] = model_id
            
        bedrock_job_request: CreateModelInvocationJobRequest = transform_openai_create_batch_to_bedrock_job_request(
            create_batch_data,
            s3_input_uri=input_s3_uri,
            s3_output_uri=output_s3_uri,
            role_arn=role_arn,
        )

        # Create boto3 client (respects env/profile/role)
        boto_client = init_bedrock_client()
        resp = boto_client.create_model_invocation_job(**bedrock_job_request)  # type: ignore
        job_arn = resp.get("jobArn")

        return LiteLLMBatch(id=job_arn or "", status="in_progress")
