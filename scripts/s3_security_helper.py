"""
S3 Security Helper Module

Provides utilities for secure S3 operations with bucket ownership verification.
Addresses security vulnerability S7608: Missing ExpectedBucketOwner parameter.

Usage:
    from scripts.s3_security_helper import s3_with_verification
    
    result = s3_with_verification(
        'list_objects_v2',
        Bucket='my-bucket',
        Prefix='data/',
        ExpectedBucketOwner='123456789012'
    )
"""

import boto3
import logging
from typing import Any, Dict, Optional
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Security constant - verified bucket owner (AWS Account ID)
# Set via environment variable: EMAILOPS_S3_EXPECTED_OWNER_ID
EXPECTED_OWNER_ID: Optional[str] = None


def set_expected_owner(account_id: str) -> None:
    """
    Set the expected S3 bucket owner (AWS Account ID).
    
    Args:
        account_id: AWS Account ID (12-digit number)
        
    Raises:
        ValueError: If account_id is not a valid 12-digit number
    """
    global EXPECTED_OWNER_ID
    
    if not isinstance(account_id, str) or not account_id.isdigit() or len(account_id) != 12:
        raise ValueError(f"Invalid AWS Account ID: {account_id}. Must be 12-digit number.")
    
    EXPECTED_OWNER_ID = account_id
    logger.info(f"Set expected S3 bucket owner: {account_id}")


def verify_bucket_ownership(s3_client: Any, bucket_name: str, account_id: str) -> bool:
    """
    Verify that a bucket is owned by the expected account.
    
    Args:
        s3_client: boto3 S3 client
        bucket_name: Name of the S3 bucket
        account_id: Expected AWS Account ID
        
    Returns:
        True if bucket owner matches, False otherwise
        
    Raises:
        ClientError: If unable to determine bucket owner
    """
    try:
        response = s3_client.head_bucket(Bucket=bucket_name)
        # The Account ID that owns the bucket is stored in response metadata
        # We verify through ACL ownership checks
        acl_response = s3_client.get_bucket_acl(Bucket=bucket_name)
        owner_id = acl_response['Owner']['ID']
        
        if owner_id != account_id:
            logger.warning(
                f"Bucket ownership mismatch: {bucket_name} owned by {owner_id}, "
                f"expected {account_id}"
            )
            return False
        
        logger.debug(f"Bucket ownership verified: {bucket_name} -> {owner_id}")
        return True
        
    except ClientError as e:
        logger.error(f"Failed to verify bucket ownership: {e}")
        raise


def s3_with_verification(
    operation: str,
    expected_owner: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Safely execute S3 operations with bucket ownership verification.
    
    This wrapper ensures the ExpectedBucketOwner parameter is used for all
    S3 operations that support it, preventing accidental bucket name confusion attacks.
    
    Args:
        operation: S3 client operation name (e.g., 'list_objects_v2', 'get_object')
        expected_owner: Expected bucket owner account ID. Defaults to EXPECTED_OWNER_ID global
        **kwargs: Arguments to pass to the S3 operation
        
    Returns:
        Operation result
        
    Raises:
        ValueError: If expected_owner not set and bucket ownership verification fails
        ClientError: If S3 operation fails
        
    Examples:
        # List objects with ownership verification
        result = s3_with_verification(
            'list_objects_v2',
            Bucket='my-bucket',
            Prefix='data/',
            expected_owner='123456789012'
        )
        
        # Get object with ownership verification
        result = s3_with_verification(
            'get_object',
            Bucket='my-bucket',
            Key='file.txt',
            expected_owner='123456789012'
        )
    """
    # Determine which owner ID to use
    owner_id = expected_owner or EXPECTED_OWNER_ID
    
    if not owner_id:
        raise ValueError(
            "ExpectedBucketOwner not set. Set via set_expected_owner() or "
            "pass expected_owner parameter to s3_with_verification()"
        )
    
    # Add ExpectedBucketOwner to operation parameters
    # Most S3 operations support this parameter
    if 'ExpectedBucketOwner' not in kwargs:
        kwargs['ExpectedBucketOwner'] = owner_id
    
    # Create S3 client
    s3_client = boto3.client('s3')
    
    try:
        # Execute the operation with verification parameter
        s3_op = getattr(s3_client, operation)
        result = s3_op(**kwargs)
        
        logger.debug(f"S3 operation {operation} completed with ownership verification")
        return result
        
    except AttributeError:
        raise ValueError(f"Unknown S3 operation: {operation}")
    except ClientError as e:
        logger.error(f"S3 operation {operation} failed: {e}")
        raise


# List of S3 operations that support ExpectedBucketOwner parameter
OPERATIONS_WITH_OWNER_CHECK = {
    # Object operations
    'get_object',
    'head_object',
    'put_object',
    'delete_object',
    'copy_object',
    'list_objects_v2',
    'list_objects',
    'get_object_acl',
    'put_object_acl',
    'get_object_tagging',
    'put_object_tagging',
    'delete_object_tagging',
    'get_object_torrent',
    'restore_object',
    'select_object_content',
    
    # Bucket operations
    'head_bucket',
    'get_bucket_acl',
    'put_bucket_acl',
    'get_bucket_tagging',
    'put_bucket_tagging',
    'delete_bucket_tagging',
    'get_bucket_cors',
    'put_bucket_cors',
    'delete_bucket_cors',
    'get_bucket_website',
    'put_bucket_website',
    'delete_bucket_website',
    'list_bucket_metrics_configurations',
    'put_bucket_metrics_configuration',
    'get_bucket_metrics_configuration',
    'delete_bucket_metrics_configuration',
    'list_bucket_inventory_configurations',
    'put_bucket_inventory_configuration',
    'get_bucket_inventory_configuration',
    'delete_bucket_inventory_configuration',
    'list_bucket_analytics_configurations',
    'put_bucket_analytics_configuration',
    'get_bucket_analytics_configuration',
    'delete_bucket_analytics_configuration',
    'get_bucket_requestpayment',
    'put_bucket_requestpayment',
    'get_bucket_versioning',
    'put_bucket_versioning',
    'delete_object_versions',
    'list_object_versions',
}


def validate_operation_supports_owner_check(operation: str) -> bool:
    """
    Check if an S3 operation supports the ExpectedBucketOwner parameter.
    
    Args:
        operation: S3 operation name
        
    Returns:
        True if operation supports ExpectedBucketOwner, False otherwise
    """
    return operation in OPERATIONS_WITH_OWNER_CHECK
