"""
S3 helpers used throughout the pipeline.

All functions accept an optional profile name. When running on EC2 with the
hackathon-ec2-profile instance profile attached, omit --profile and credentials
come from the instance metadata service automatically.

Default bucket: scrippsresearch-grotjahn-hackathon  (set GROTJAHN_S3_BUCKET env var to override)
Default region: us-west-2
"""

import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

DEFAULT_BUCKET = os.environ.get("GROTJAHN_S3_BUCKET", "scrippsresearch-grotjahn-hackathon")
DEFAULT_REGION = "us-west-2"
DEFAULT_PROFILE = os.environ.get("AWS_PROFILE", None)   # None → use instance profile on EC2


def _s3(profile: str | None = DEFAULT_PROFILE) -> boto3.client:
    if profile:
        session = boto3.Session(profile_name=profile, region_name=DEFAULT_REGION)
    else:
        session = boto3.Session(region_name=DEFAULT_REGION)
    return session.client("s3")


def ensure_bucket(bucket: str = DEFAULT_BUCKET, profile: str | None = DEFAULT_PROFILE):
    """Create the bucket if it doesn't exist."""
    s3 = _s3(profile)
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": DEFAULT_REGION},
            )
            print(f"Created bucket: s3://{bucket}/")
        else:
            raise


def upload(local_path: str | Path, s3_key: str,
           bucket: str = DEFAULT_BUCKET, profile: str | None = DEFAULT_PROFILE):
    """Upload a local file to S3."""
    s3 = _s3(profile)
    local_path = Path(local_path)
    s3.upload_file(str(local_path), bucket, s3_key)
    print(f"  ↑ s3://{bucket}/{s3_key}")


def download(s3_key: str, local_path: str | Path,
             bucket: str = DEFAULT_BUCKET, profile: str | None = DEFAULT_PROFILE):
    """Download a file from S3, creating parent directories as needed."""
    s3 = _s3(profile)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, s3_key, str(local_path))
    print(f"  ↓ s3://{bucket}/{s3_key}  →  {local_path}")


def sync_up(local_dir: str | Path, s3_prefix: str,
            bucket: str = DEFAULT_BUCKET, profile: str | None = DEFAULT_PROFILE,
            extensions: tuple = (".npy", ".npz", ".pkl", ".csv", ".yaml", ".txt", ".png")):
    """Upload all matching files in local_dir to s3://bucket/s3_prefix/."""
    local_dir = Path(local_dir)
    uploaded = 0
    for f in sorted(local_dir.rglob("*")):
        if f.is_file() and (not extensions or f.suffix in extensions):
            rel = f.relative_to(local_dir)
            key = f"{s3_prefix.rstrip('/')}/{rel}"
            upload(f, key, bucket=bucket, profile=profile)
            uploaded += 1
    print(f"Synced {uploaded} file(s) → s3://{bucket}/{s3_prefix}/")
    return uploaded


def sync_down(s3_prefix: str, local_dir: str | Path,
              bucket: str = DEFAULT_BUCKET, profile: str | None = DEFAULT_PROFILE):
    """Download all objects under s3_prefix into local_dir."""
    s3 = _s3(profile)
    local_dir = Path(local_dir)
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(s3_prefix):].lstrip("/")
            if rel:
                local_path = local_dir / rel
                download(key, local_path, bucket=bucket, profile=profile)
                downloaded += 1
    print(f"Downloaded {downloaded} file(s) ← s3://{bucket}/{s3_prefix}/")
    return downloaded


def list_keys(s3_prefix: str, bucket: str = DEFAULT_BUCKET,
              profile: str | None = DEFAULT_PROFILE) -> list[str]:
    s3 = _s3(profile)
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="S3 sync helper")
    sub = parser.add_subparsers(dest="cmd")

    up = sub.add_parser("up", help="Sync local dir to S3")
    up.add_argument("local_dir")
    up.add_argument("s3_prefix")
    up.add_argument("--bucket", default=DEFAULT_BUCKET)
    up.add_argument("--profile", default=DEFAULT_PROFILE)

    dn = sub.add_parser("down", help="Sync S3 prefix to local dir")
    dn.add_argument("s3_prefix")
    dn.add_argument("local_dir")
    dn.add_argument("--bucket", default=DEFAULT_BUCKET)
    dn.add_argument("--profile", default=DEFAULT_PROFILE)

    mb = sub.add_parser("mb", help="Create bucket if needed")
    mb.add_argument("--bucket", default=DEFAULT_BUCKET)
    mb.add_argument("--profile", default=DEFAULT_PROFILE)

    args = parser.parse_args()
    if args.cmd == "up":
        ensure_bucket(args.bucket, args.profile)
        sync_up(args.local_dir, args.s3_prefix, bucket=args.bucket, profile=args.profile)
    elif args.cmd == "down":
        sync_down(args.s3_prefix, args.local_dir, bucket=args.bucket, profile=args.profile)
    elif args.cmd == "mb":
        ensure_bucket(args.bucket, args.profile)
    else:
        parser.print_help()
