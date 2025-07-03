#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
import time

# Third-party imports
import boto3
import docx
from docx import Document
import PyPDF2
import markdown
from bs4 import BeautifulSoup
from botocore.config import Config
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def extract_code_blocks_from_readme(readme_path):
    code_blocks = []
    current_file = None
    current_code = []
    in_code_block = False

    with open(readme_path, encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        header_match = re.match(r"^### [`'](.+?)['`]", line)
        if header_match:
            current_file = header_match.group(1)
            continue

        if line.strip().startswith("```") and current_file and not in_code_block:
            in_code_block = True
            current_code = []
            continue

        if line.strip() == "```" and in_code_block:
            if current_file and current_code:
                code_blocks.append((current_file, "".join(current_code)))
            in_code_block = False
            current_file = None
            current_code = []
            continue

        if in_code_block and current_file:
            current_code.append(line)

    return code_blocks

def create_files_from_blocks(code_blocks, base_dir="."):
    for filepath, code in code_blocks:
        full_path = os.path.join(base_dir, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Created: {full_path}")

def extract_code_blocks_from_response(generated_text: str) -> Dict[str, str]:
    files = {}
    current_file = None
    current_content = []
    in_code_block = False

    lines = generated_text.split('\n')
    for line in lines:
        if line.startswith('```') and not in_code_block:
            parts = line.strip('`').split(' ', 1)
            if len(parts) > 1:
                current_file = parts[1].strip()
            in_code_block = True
            current_content = []
            continue
        if line.strip() == '```' and in_code_block:
            if current_file and current_content:
                files[current_file] = '\n'.join(current_content)
            current_file = None
            in_code_block = False
            continue
        if in_code_block and current_file:
            current_content.append(line)
    return files

class DocumentParser:
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.content = ""

    def extract_content(self) -> str:
        raise NotImplementedError("Subclasses must implement extract_content")

class DocxParser(DocumentParser):
    def extract_content(self) -> str:
        doc = docx.Document(self.file_path)
        return "\n".join([para.text for para in doc.paragraphs])

class PdfParser(DocumentParser):
    def extract_content(self) -> str:
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages])

class MarkdownParser(DocumentParser):
    def extract_content(self) -> str:
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return file.read()

class BedrockClient:
    def __init__(self, model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0.0,
                 region="us-east-1", aws_access_key=None, aws_secret_key=None):
        config = Config(connect_timeout=600, read_timeout=600)
        if aws_access_key and aws_secret_key:
            self.bedrock_runtime = boto3.client(
                'bedrock-runtime', region_name=region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                config=config)
        else:
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region, config=config)
        self.model_id = model_id
        self.temperature = temperature

    def generate_code(self, document_content: str, system_prompt=None, max_tokens=64000) -> str:
        attempt = 0
        max_attempts = 3
        continuation_limit = 5

        prompt = self._build_prompt(document_content)
        while attempt < max_attempts:
            try:
                initial_response = self._invoke_streaming(prompt, max_tokens, system_prompt)
                response = initial_response
                continuation_count = 0
                while self._is_response_incomplete(response) and continuation_count < continuation_limit:
                    continuation_prompt = self._build_continuation_prompt(response, document_content)
                    continuation = self._invoke_streaming(continuation_prompt, max_tokens, system_prompt)
                    response = self._merge_responses(response, continuation)
                    continuation_count += 1
                return response
            except Exception as e:
                attempt += 1
                time.sleep(3)
        raise Exception("All attempts to generate code failed.")

    def _build_prompt(self, doc: str) -> str:
        return f"""You are an expert Python developer. Based on the following documentation,\ngenerate well-structured Python code that follows best practices.\n\nDocumentation:\n{doc}"""

    def _build_continuation_prompt(self, prev_response: str, doc: str) -> str:
        context = prev_response[-1000:] if len(prev_response) > 1000 else prev_response
        return f"""The previous response was incomplete. Please continue from here:\n\nContext:\n{context}\n\nOriginal Documentation:\n{doc}"""

    def _is_response_incomplete(self, response: str) -> bool:
        return response.count('{') > response.count('}') or \
               response.strip().endswith(('class', 'def', '(', ':', ',')) or \
               len(response) < 2000

    def _merge_responses(self, part1: str, part2: str) -> str:
        return part1.strip() + '\n\n' + part2.strip()

    def _invoke_streaming(self, prompt: str, max_tokens: int, system_prompt: str = None) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }
        if system_prompt:
            body["system"] = system_prompt

        response = self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )

        output = ""
        for event in response['body']:
            chunk = event.get('chunk')
            if chunk:
                decoded = json.loads(chunk['bytes'].decode())
                if decoded.get('type') == 'content_block_delta':
                    delta = decoded.get('delta', {})
                    output += delta.get('text', '')
                elif decoded.get('type') == 'message_stop':
                    break
        return output

class CodeGenerator:
    def __init__(self, input_path: str, output_dir: str, profile: str,
                 aws_region: str = "us-east-1", aws_access_key: Optional[str] = None,
                 aws_secret_key: Optional[str] = None, model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.profile = profile
        self.client = BedrockClient(model_id=model_id, region=aws_region, aws_access_key=aws_access_key, aws_secret_key=aws_secret_key, temperature=0.2)
        self.supported_extensions = {'.docx', '.pdf', '.md'}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_documents(self) -> List[Path]:
        documents = []
        if self.input_path.is_file():
            if self.input_path.suffix.lower() in self.supported_extensions:
                documents.append(self.input_path)
        elif self.input_path.is_dir():
            for ext in self.supported_extensions:
                documents.extend(self.input_path.glob(f"**/*{ext}"))
        return documents

    def parse_document(self, document_path: Path) -> str:
        ext = document_path.suffix.lower()
        if ext == '.docx':
            return DocxParser(document_path).extract_content()
        elif ext == '.pdf':
            return PdfParser(document_path).extract_content()
        elif ext == '.md':
            return MarkdownParser(document_path).extract_content()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def save_generated_files(self, files: Dict[str, str]) -> None:
        for file_path, content in files.items():
            full_path = self.output_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created file: {full_path}")

    def generate(self) -> None:
        documents = self.find_documents()
        if not documents:
            logger.error(f"No documents found in {self.input_path}")
            return

        combined_content = ""
        for doc_path in documents:
            logger.info(f"Parsing document: {doc_path}")
            combined_content += f"\n\n### Document: {doc_path.name}\n\n{self.parse_document(doc_path)}"

        logger.info("Generating code with Claude Sonnet...")
        generated_text = self.client.generate_code(combined_content)
        files = extract_code_blocks_from_response(generated_text)
        if not files:
            files = {"README.md": generated_text}
        self.save_generated_files(files)

        readme_path = self.output_dir / "README.md"
        if readme_path.exists():
            self.create_files_from_readme(readme_path)

    def create_files_from_readme(self, readme_path: Path) -> None:
        logger.info(f"Extracting from README.md: {readme_path}")
        blocks = extract_code_blocks_from_readme(readme_path)
        for filepath, code in blocks:
            full_path = self.output_dir / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(code)
            logger.info(f"Created file: {full_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="Input document path")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--model", default="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    parser.add_argument("--output-dir", default="./generated_code")
    parser.add_argument("--from-readme", action="store_true")
    parser.add_argument("--readme-path", default="README.md")
    args = parser.parse_args()

    if args.from_readme:
        create_files_from_blocks(extract_code_blocks_from_readme(args.readme_path), base_dir=args.output_dir)
        sys.exit(0)

    if args.profile:
        os.environ['AWS_PROFILE'] = args.profile
        session = boto3.Session(profile_name=args.profile, region_name=args.aws_region)
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"AWS credentials OK. Account ID: {identity['Account']}")

    generator = CodeGenerator(
        input_path=args.input_dir,
        output_dir=args.output_dir,
        profile=args.profile,
        aws_region=args.aws_region,
        model_id=args.model
    )
    generator.generate()

if __name__ == "__main__":
    main()
