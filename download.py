#!/usr/bin/env python3
"""
Model downloader for llama.cpp Docker container
Downloads models from HuggingFace with environment variable configuration
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, list_repo_files
import requests

def main():
    # Load .env file if it exists
    load_dotenv()

    # Set HF cache to local directory if not set
    if not os.getenv('HF_HOME'):
        os.environ['HF_HOME'] = './hf_cache'

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download models from HuggingFace for llama.cpp')
    parser.add_argument('--repo', help='HuggingFace repository (overrides MODEL_REPO)')
    parser.add_argument('--pattern', help='File pattern to download (overrides MODEL_PATTERN)')
    parser.add_argument('--api-name', help='API name for model (overrides API_NAME)')
    parser.add_argument('--output-dir', default='models', help='Base output directory (default: models)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded without downloading')
    parser.add_argument('--list-files', action='store_true', help='List files matching pattern in repository')
    parser.add_argument('--force', action='store_true', help='Force re-download even if model exists')
    parser.add_argument('--env-file', default='.env', help='Path to .env file (default: .env)')

    args = parser.parse_args()

    # Get configuration from env vars or command line
    model_repo = args.repo or os.getenv('MODEL_REPO')
    model_pattern = args.pattern or os.getenv('MODEL_PATTERN')
    api_name = args.api_name or os.getenv('API_NAME')
    model_file = os.getenv('MODEL_FILE')  # Main model file to check existence
    hf_token = os.getenv('HF_TOKEN')  # Will be auto-used by huggingface_hub
    # Treat empty string as None to avoid 401 errors for public repos
    if hf_token == '':
        hf_token = None

    # Enable fast transfer if set
    if os.getenv('HF_HUB_ENABLE_HF_TRANSFER') == '1':
        print("Fast transfer mode enabled (hf_transfer)")

    # Validate required parameters
    if not model_repo:
        print("Error: MODEL_REPO environment variable or --repo argument is required")
        sys.exit(1)

    # List files if requested (doesn't need api_name)
    if args.list_files:
        try:
            print(f"Listing files in {model_repo}...")
            files = list_repo_files(model_repo, token=hf_token)

            # Filter by pattern if provided
            if model_pattern:
                import fnmatch
                files = [f for f in files if fnmatch.fnmatch(f, model_pattern)]

            print(f"\nFound {len(files)} matching files:")
            for f in sorted(files):
                print(f"  {f}")
        except Exception as e:
            print(f"Error listing files: {e}")
            sys.exit(1)
        return

    # API name required for downloads
    if not api_name:
        print("Error: API_NAME environment variable or --api-name argument is required")
        sys.exit(1)

    # Determine output directory
    output_path = Path(args.output_dir) / api_name

    print(f"Configuration:")
    print(f"  Repository: {model_repo}")
    print(f"  Pattern: {model_pattern or 'all files'}")
    print(f"  API Name: {api_name}")
    print(f"  Output Directory: {output_path}")
    if model_file:
        print(f"  Main Model File: {model_file}")
    print()

    # Check if model already exists (if final directory exists, download is complete)
    if not args.force and output_path.exists():
        print(f"Model already exists at {output_path}")
        print("Use --force to re-download")
        return

    # Dry run mode
    if args.dry_run:
        print("DRY RUN MODE - Would download:")
        print(f"  From: {model_repo}")
        print(f"  Pattern: {model_pattern or 'all files'}")
        print(f"  To: {output_path}")
        return

    # Use atomic download approach
    temp_path = output_path.with_suffix('.downloading')
    final_path = output_path

    # Create temp directory for atomic download
    temp_path.mkdir(parents=True, exist_ok=True)

    # Download the model to temp location
    try:
        print(f"Downloading model from {model_repo} to temporary location...")
        print(f"Temp path: {temp_path}")

        snapshot_download(
            repo_id=model_repo,
            local_dir=str(temp_path),
            allow_patterns=model_pattern.split(',') if model_pattern else None,
            token=hf_token
        )

        print(f"\nDownload successful! Verifying and moving to final location...")

        # Verify main model file exists if specified
        if model_file:
            model_file_path = temp_path / model_file
            if not model_file_path.exists():
                print(f"Error: Expected model file {model_file} not found after download!")
                print("Available files in temp directory:")
                for f in temp_path.glob("*"):
                    if f.is_file():
                        print(f"  {f.name} ({f.stat().st_size} bytes)")
                print("Cleaning up temp directory...")
                import shutil
                shutil.rmtree(temp_path)
                sys.exit(1)
            else:
                print(f"Verified main model file: {model_file_path}")

        # Atomic move: remove old directory if it exists, then move temp to final
        if final_path.exists():
            print(f"Removing incomplete download at {final_path}")
            import shutil
            shutil.rmtree(final_path)

        print(f"Moving complete download from {temp_path} to {final_path}")
        temp_path.rename(final_path)

        print(f"\nDownload complete! Model saved to: {final_path}")

    except Exception as e:
        print(f"Error downloading model: {e}")
        # Clean up temp directory on failure
        if temp_path.exists():
            print(f"Cleaning up temporary directory: {temp_path}")
            import shutil
            shutil.rmtree(temp_path)
        sys.exit(1)

if __name__ == "__main__":
    main()
