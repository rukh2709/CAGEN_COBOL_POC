
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

# Third-party imports (you'll need to install these)
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
    """
    Extract code blocks and their file paths from the README.md.
    Returns a list of (filepath, code) tuples.
    """
    code_blocks = []
    current_file = None
    current_code = []
    in_code_block = False

    with open(readme_path, encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Look for a header like ### `src/config.py`
        header_match = re.match(r"^### [`'](.+?)['`]", line)
        if header_match:
            current_file = header_match.group(1)
            continue

        # Start of code block
        if line.strip().startswith("```") and current_file and not in_code_block:
            in_code_block = True
            current_code = []
            continue

        # End of code block
        if line.strip() == "```" and in_code_block:
            if current_file and current_code:
                code_blocks.append((current_file, "".join(current_code)))
            in_code_block = False
            current_file = None
            current_code = []
            continue

        # Collect code lines
        if in_code_block and current_file:
            current_code.append(line)

    return code_blocks

def create_files_from_blocks(code_blocks, base_dir="."):
    """
    Create files and directories from the extracted code blocks.
    """
    for filepath, code in code_blocks:
        full_path = os.path.join(base_dir, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Created: {full_path}")

def extract_code_blocks_from_response(generated_text: str) -> Dict[str, str]:
    """
    Extract file paths and code from Claude response in markdown format:
    ```python filename.py
    code here
    ```
    """
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
    """Base class for document parsing functionality."""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.content = ""
        
    def extract_content(self) -> str:
        """Extract content from the document. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement extract_content")


class DocxParser(DocumentParser):
    """Parser for Microsoft Word documents (.docx)."""
    
    def extract_content(self) -> str:
        try:
            doc = docx.Document(self.file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
            return content
        except Exception as e:
            logger.error(f"Error extracting content from DOCX: {e}")
            raise


class PdfParser(DocumentParser):
    """Parser for PDF documents."""
    
    def extract_content(self) -> str:
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            return content
        except Exception as e:
            logger.error(f"Error extracting content from PDF: {e}")
            raise


class MarkdownParser(DocumentParser):
    """Parser for Markdown documents."""
    
    def extract_content(self) -> str:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            logger.error(f"Error extracting content from Markdown: {e}")
            raise


class BedrockClient:
    
    """Client for communicating with claude sonnet via AWS Bedrock."""
    
    def __init__(self, model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0.0, 
                 region="us-east-1", aws_access_key=None, aws_secret_key=None):
        config = Config(connect_timeout=600, read_timeout=600)
        try:
            if aws_access_key and aws_secret_key:
                logger.info("Using provided AWS access key and secret key")
                self.bedrock_runtime = boto3.client(
                    'bedrock-runtime',
                    region_name=region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    config=config)
            else:
                logger.info("Attempting to use AWS credentials from environment or credentials file")
                self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region, config=config)
            if not self.bedrock_runtime:
                raise ValueError("Failed to initialize Bedrock client")
            self.model_id = model_id
            self.temperature = temperature
            logger.info(f"Initialized Bedrock client with model {model_id} in region {region}")
        except Exception as e:
            logger.error(f"Bedrock initialization error: {str(e)}")
            raise ValueError(f"Failed to initialize Bedrock client: {str(e)}")
            
    def generate_code(self, document_content: str, system_prompt=None, max_tokens=64000) -> str:
        """
        Generate Python code from document content using Claude Sonnet.
        
        Args:
            document_content: The extracted text from the documentation
            
        Returns:
            Generated Python code
        """
        try:
            
            
            prompt = f"""
You are an expert Python developer. Based on the following documentation, 
generate well-structured Python code that follows OOP principles, has proper folder 
and file structure, includes thorough error handling, and implements the described functionality.

The code should include:
1. Proper class hierarchies based on the business domain
2. Clear separation of concerns (data access, business logic, UI/presentation)
3. Comprehensive exception handling
4. Type hints and documentation
5. Unit test structure
6. Requirements.txt file contents
7. README.md with installation and usage instructions displaying the tree structure of the project. 
8. For every file in the project, output a header in the format ### 'relative/path/to/file.py' followed by a code block with the file contents.
9. Create the sample input data file if applicable.
Documentation:
s
{document_content}

Please provide a complete implementation with all necessary files and their contents
structured according to best practices.
"""
            # Prepare the messages for the Bedrock API
            
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            # Format the payload according to Bedrock API requirements    
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "messages": messages
                }

            # If system prompt is provided, add it properly as a system message
            if system_prompt:
                # For Claude on Bedrock, we need to use a special format
                if "anthropic.claude" in self.model_id:
                    request_body["system"] = system_prompt
                else:
                    # For other models that support system messages differently
                    logger.warning("System prompts may not be supported in this model format")

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
                  
            )
            
            response_body = json.loads(response.get('body').read())
            generated_code = response_body['content'][0]['text']
            return generated_code
            
        except Exception as e:
            logger.error(f"Error generating code with Claude: {e}")
            raise

class CodeGenerator:
    """Main class for generating Python code from documentation."""
    
    def __init__(self, input_path: str, output_dir: str, profile: str , aws_region: str = "us-east-1", 
                 aws_access_key: Optional[str] = None, 
                 aws_secret_key: Optional[str] = None, model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
        
        self.input_path = Path(input_path)
        self.profile = profile
        self.aws_region = aws_region
        self.output_dir = Path(output_dir)
        aws_access_key=aws_access_key
        aws_secret_key=aws_secret_key
        temperature=0.2
        self.supported_extensions = {'.docx', '.pdf', '.md'}
        
        # Initialize Bedrock client
        try:
            self.client = BedrockClient(
                model_id=model_id,
                region=aws_region,
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                temperature=0.2
            )
            # Test connection with a small request
            # self.client.generate_code("Say OK", max_tokens=10)
            # logger.info("Successfully connected to Amazon Bedrock")
        except Exception as e:
            raise ValueError(f"Failed to initialize Bedrock client: {str(e)}\nPlease check your AWS credentials and permissions.")
       
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        
        
    
    def find_documents(self) -> List[Path]:
        """
        Find all supported document files in the input path.
        If input_path is a file, return it directly if supported.
        If input_path is a directory, search recursively for supported files.
        
        Returns:
            List of paths to supported documents
        """
        documents = []
        
        if self.input_path.is_file():
            if self.input_path.suffix.lower() in self.supported_extensions:
                documents.append(self.input_path)
            else:
                logger.warning(f"Unsupported file format: {self.input_path}")
        
        elif self.input_path.is_dir():
            for ext in self.supported_extensions:
                # Find all files with the supported extension in the directory and subdirectories
                for file_path in self.input_path.glob(f"**/*{ext}"):
                    documents.append(file_path)
        
        else:
            raise ValueError(f"Input path does not exist: {self.input_path}")
        
        logger.info(f"Found {len(documents)} documents to process")
        return documents
        
    def parse_document(self, document_path: Path) -> str:
        """Parse a document based on its file extension."""
        file_extension = document_path.suffix.lower()
        
        parser: DocumentParser
        if file_extension == '.docx':
            parser = DocxParser(document_path)
        elif file_extension == '.pdf':
            parser = PdfParser(document_path)
        elif file_extension == '.md':
            parser = MarkdownParser(document_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return parser.extract_content()
    
        
    def save_generated_files(self, files: Dict[str, str]) -> None:
        """
        Save the generated files to the output directory.
        
        Args:
            files: Dictionary mapping file paths to their content
        """
        for file_path, content in files.items():
            # Create the full path
            full_path = self.output_dir / file_path
            
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file content
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Created file: {full_path}")
            except Exception as e:
                logger.error(f"Error creating file {full_path}: {e}")
    
    def generate(self) -> None:
        """
        Main method to generate Python code from the input documents.
        """
        try:
            # Find all documents to process
            documents = self.find_documents()
            
            if not documents:
                logger.error(f"No supported documents found in {self.input_path}")
                return
            
            # Extract content from all documents
            combined_content = ""
            for doc_path in documents:
                logger.info(f"Parsing document: {doc_path}")
                try:
                    doc_content = self.parse_document(doc_path)
                    combined_content += f"\n\n### Document: {doc_path.name}\n\n{doc_content}"
                except Exception as e:
                    logger.error(f"Error parsing document {doc_path}: {e}")
                    logger.error("Skipping this document and continuing with others")
            
            if not combined_content:
                logger.error("No content extracted from any documents")
                return
            
            logger.info("Generating code with Claude Sonnet...")
            generated_text = self.client.generate_code(combined_content, max_tokens=64000)
            
            logger.info("Extracting code blocks...")
            files = extract_code_blocks_from_response(generated_text)
            
            if not files:
                # If no structured code blocks were found, treat the entire response as README
                files = {"README.md": generated_text}
            
            logger.info(f"Saving {len(files)} generated files...")
            self.save_generated_files(files)

            

            # If README.md was one of the generated files, create components from it
            readme_path = self.output_dir / "README.md"
            if readme_path.exists():
                # Save a copy as README.md_generated
                generated_readme_copy = self.output_dir / "README_generated.md"
                try:
                    with open(readme_path, "r", encoding="utf-8") as src, open(generated_readme_copy, "w", encoding="utf-8") as dst:
                        dst.write(src.read())
                    logger.info(f"Copied generated README.md to: {generated_readme_copy}")
                except Exception as e:
                    logger.error(f"Failed to copy README.md: {e}")

                logger.info("README.md found — extracting components and generating files...")
                time.sleep(5)
                self.create_files_from_readme(readme_path)

            logger.info(f"Code generation complete. Output saved to: {self.output_dir}")
                                   
        except Exception as e:
            logger.error(f"Error during code generation: {e}")
            raise
    
    def create_files_from_readme(self, readme_path: Path) -> None:
        """
        Parses the README.md file and generates files from code blocks.
        """
        try:
            logger.info(f"Parsing README.md at: {readme_path}")
            code_blocks = extract_code_blocks_from_readme(readme_path)
            if not code_blocks:
                logger.warning("No code blocks with file paths found in README.md.")
                return

            for filepath, code in code_blocks:
                full_path = self.output_dir / filepath
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(code)
                logger.info(f"Created file from README.md: {full_path}")

        except Exception as e:
            logger.error(f"Error while generating files from README.md: {e}")


def main():
    """
    Main entry point for the code generator CLI.
    """
    parser = argparse.ArgumentParser(description="Generate Python code from documentation using Claude Sonnet")
    
    parser.add_argument(
        "--input-dir", 
        help="Path to the input document file or directory containing documents (docx, pdf, md)"
    )
    
    parser.add_argument(
        "--profile", 
        help="AWS profile name to use from credentials file")

    
    parser.add_argument(
        "--aws-region", "-r",
        default="us-east-1",
        help="AWS region for Bedrock service (default: us-east-1)"
    )

    parser.add_argument(
        "--model", 
        default="us.anthropic.claude-3-7-sonnet-20250219-v1:0", 
        help="Claude model ID to use in Bedrock"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="./generated_code",
        help="Directory where generated code will be saved (default: ./generated_code)"
    )

    parser.add_argument(
    "--from-readme",
    action="store_true",
    help="If set, use the given README.md to create the folder structure and files."
    )
    parser.add_argument(
        "--readme-path",
        default="README.md",
        help="Path to the README.md file to use for folder structure creation."
    )
    
    args = parser.parse_args()

    if args.from_readme:
        code_blocks = extract_code_blocks_from_readme(args.readme_path)
        if not code_blocks:
            print("No code blocks with file paths found in README.md.")
            sys.exit(1)
        create_files_from_blocks(code_blocks, base_dir=args.output_dir)
        print("All files and folders created as per README.md.")
        sys.exit(0)

    logger.info("Done.")

    try:
        # Check AWS credentials before initializing the generator
        if args.profile:
            # Use the specified profile
            os.environ['AWS_PROFILE'] = args.profile
            logger.info(f"Using AWS profile: {args.profile}")
 
            # Verify AWS credentials are accessible
            try:
                session = boto3.Session(
                    profile_name=args.profile,
                    region_name=args.aws_region
                )
               
                # Test if credentials are valid by making a simple API call
                sts = session.client('sts')
                identity = sts.get_caller_identity()
                logger.info(f"AWS credentials verified. Account ID: {identity['Account']}")
 
            except Exception as e:
                logger.error(f"AWS credential verification failed: {str(e)}")
                logger.error("Please check your AWS credentials. You can:")
                logger.error("1. Set up credentials file at ~/.aws/credentials")
                logger.error("2. Provide --aws-access-key and --aws-secret-key arguments")
                logger.error("3. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
                logger.error("4. Specify an AWS profile with --profile")
                sys.exit(1)
        
   
        generator = CodeGenerator(
            input_path=args.input_dir,
            output_dir=args.output_dir,
            profile=args.profile,
            aws_region=args.aws_region
        )
        generator.generate()
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
