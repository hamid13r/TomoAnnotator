"""
Generate a human-readable report from classification results using Claude on Bedrock.

Reads results/predictions.csv (output of classify.py) and asks Claude to:
  - Summarize which features are most common
  - Flag the most interesting tomograms to look at first
  - Note any patterns (e.g., tomograms with both mito AND ER might show contact sites)

Claude runs on Bedrock in us-west-2 using the hackathon instance profile.
Works from EC2 (instance profile credentials) or laptop (SSO profile).

Usage:
    python report.py --predictions results/predictions.csv
    python report.py --predictions results/predictions.csv --model sonnet
    python report.py --predictions results/predictions.csv --push-s3
"""

import argparse
import json
from pathlib import Path

import boto3
import pandas as pd

MODELS = {
    "haiku":  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "opus":   "us.anthropic.claude-opus-4-6-v1",
}


def load_predictions(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    # Feature columns are those without _conf suffix and not tomogram_id
    feature_cols = [c for c in df.columns
                    if c != "tomogram_id" and not c.endswith("_conf")]
    return df, feature_cols


def build_prompt(df: pd.DataFrame, feature_cols: list[str]) -> str:
    lines = ["tomogram_id," + ",".join(f"{c}(confidence)" for c in feature_cols)]
    for _, row in df.iterrows():
        cells = []
        for f in feature_cols:
            present = row.get(f, "?")
            conf = row.get(f"{f}_conf", "?")
            cells.append(f"{present}({conf})")
        lines.append(f"{row.tomogram_id}," + ",".join(cells))
    table = "\n".join(lines)

    return f"""You are analyzing cryo-electron tomography (cryoET) data from the Grotjahn Lab at Scripps Research Institute. The lab studies mitochondrial morphology and cellular organelle ultrastructure.

A machine learning classifier has analyzed a collection of cellular tomograms and predicted which organelles and cellular features are present in each one. The results are below.

Features flagged:
- mitochondria: mitochondria outer membrane visible
- er: endoplasmic reticulum membrane network
- microtubules: microtubule filaments
- vesicles: membrane-bound vesicles
- ribosomes: ribosome particles
- nuclear_envelope: nuclear envelope

Classification results (YES/no = predicted presence, value in parentheses = confidence 0-1):

{table}

Please provide:
1. **Summary**: Which features are most commonly present across the dataset? Which are rare?
2. **Top 3 tomograms to examine first**: Which tomograms are most interesting and why? Consider tomograms that have multiple features (e.g., both mitochondria and ER could show mitochondrial-ER contact sites), or rare features worth investigating.
3. **Dataset patterns**: Any notable patterns — e.g., do certain features tend to co-occur? Are any tomograms unusual?
4. **Caveats**: Flag any predictions with low confidence (<0.6) that should be manually verified.

Keep the response concise and actionable for a biologist doing a first-pass review of their tomogram library.
"""


def call_bedrock(prompt: str, model_id: str, profile: str | None = None) -> str:
    if profile:
        session = boto3.Session(profile_name=profile, region_name="us-west-2")
    else:
        session = boto3.Session(region_name="us-west-2")

    bedrock = session.client("bedrock-runtime")

    resp = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    return json.loads(resp["body"].read())["content"][0]["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="results/predictions.csv")
    parser.add_argument("--model", default="haiku", choices=list(MODELS),
                        help="Claude model to use (haiku=fast/cheap, sonnet=better)")
    parser.add_argument("--output", default=None, help="Save report to .md file")
    parser.add_argument("--profile", default=None,
                        help="AWS SSO profile (omit on EC2 with instance profile)")
    parser.add_argument("--push-s3", action="store_true",
                        help="Upload report to S3 after generating")
    parser.add_argument("--bucket", default=None, help="S3 bucket (overrides default)")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"Predictions file not found: {pred_path}")
        print("Run classify.py first.")
        return

    df, feature_cols = load_predictions(pred_path)
    print(f"Loaded {len(df)} tomograms with features: {feature_cols}")
    print(f"Calling Claude {args.model} on Bedrock (us-west-2)...")

    prompt = build_prompt(df, feature_cols)
    model_id = MODELS[args.model]

    try:
        report = call_bedrock(prompt, model_id, profile=args.profile)
    except Exception as e:
        print(f"Bedrock call failed: {e}")
        print("If on laptop, pass --profile <your-sso-profile>")
        print("If token expired, run: aws sso login --profile <profile>")
        return

    print("\n" + "="*60)
    print(report)
    print("="*60)

    out_path = Path(args.output) if args.output else pred_path.parent / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"# Tomogram Feature Report\n\nGenerated from: `{pred_path}`  \nModel: Claude {args.model}\n\n"
    out_path.write_text(header + report)
    print(f"\nSaved: {out_path}")

    if args.push_s3:
        from aws_utils import ensure_bucket, upload, DEFAULT_BUCKET
        bucket = args.bucket or DEFAULT_BUCKET
        ensure_bucket(bucket, args.profile)
        upload(out_path, f"results/{out_path.name}", bucket=bucket, profile=args.profile)
        upload(pred_path, f"results/{pred_path.name}", bucket=bucket, profile=args.profile)


if __name__ == "__main__":
    main()
