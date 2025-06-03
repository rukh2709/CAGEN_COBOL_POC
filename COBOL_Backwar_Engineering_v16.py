import os
import sys
import re
import time
import json
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Set, Tuple, Any, Optional
import concurrent.futures
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cobol_documentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import boto3, provide helpful error if missing
try:
    import boto3
    from botocore.config import Config
except ImportError:
    logger.error("The 'boto3' package is not installed. Please install it with: pip install boto3")
    sys.exit(1)

class BedrockClient:
    """Client for interacting with Amazon Bedrock API."""
    
    def __init__(self, model_id="anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0.0, 
                 region="us-east-1", aws_access_key=None, aws_secret_key=None):
        """Initialize the Bedrock client."""
        
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
                # Try to load credentials from environment or credentials file
                logger.info("Attempting to use AWS credentials from environment or credentials file")
                self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region, config=config)

            # Fixed: Replace the invalid validation method with a simple check
            if not self.bedrock_runtime:
                raise ValueError("Failed to initialize Bedrock client")

            self.model_id = model_id
            self.temperature = temperature
            logger.info(f"Initialized Bedrock client with model {model_id} in region {region}")
        
        except Exception as e:
            logger.error(f"Bedrock initialization error: {str(e)}")
            raise ValueError(f"Failed to initialize Bedrock client: {str(e)}")
    
    def generate_text(self, prompt, system_prompt=None, max_tokens=64000):
        """Generate text using Claude through Amazon Bedrock."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        try:
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
            # Extract token usage from the response body
            usage = response_body.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
        
            if 'content' in response_body and len(response_body['content']) > 0:
                # Extract the text from the first content block
                output_text = response_body['content'][0]['text']                
              
                logger.debug(f"Response Body: {response_body}")
                
                # Save detailed information to a JSON file
                log_data = {
                    "system_prompt": system_prompt,
                    "base_prompt": prompt,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "response_body": response_body
                }
                
                # Create a directory for logs if it doesn't exist
                log_dir = "logs"
                os.makedirs(log_dir, exist_ok=True)
                
                # Save the log with a timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"log_{timestamp}.json")
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=4)
                
                return output_text
            else:
                logger.error(f"Unexpected response structure: {response_body}")
                return "Error: Unexpected response format from Claude"

        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}")
            if "ValidationException" in str(e):
                logger.error("This appears to be a formatting issue with the request to Bedrock.")
                logger.error("Input prompt length: " + str(len(prompt)))
            raise e

    def stream_generate_text(self, prompt, system_prompt=None, max_tokens=64000):
        """Generate text using Claude through Amazon Bedrock with streaming response."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
            
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        response = self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(request_body)
        )
        
        stream = response.get('body')
        full_response = ""
        
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_data = json.loads(chunk.get('bytes').decode())
                if chunk_data.get('type') == 'content_block_delta' and chunk_data.get('delta', {}).get('type') == 'text':
                    text_chunk = chunk_data.get('delta', {}).get('text', '')
                    full_response += text_chunk
                    yield text_chunk
        
        return full_response


class CobolDocumentationGenerator:
    """Main class for generating COBOL documentation using Claude AI."""
    
    def __init__(self, folder_path: str, 
                 model_id: str = "anthropic.claude-3-7-sonnet-20250219-v1:0", 
                 region: str = "us-east-1",
                 aws_access_key: Optional[str] = None, 
                 aws_secret_key: Optional[str] = None,
                 max_chunk_size: int = 80000, 
                 max_workers: int = 5,
                 output_dir: str = "documentation", 
                 ignore_dirs: List[str] = None):
        """
        Initialize the COBOL documentation generator.
        
        Args:
            folder_path: Path to the folder containing COBOL files
            model_id: Claude model ID to use in Amazon Bedrock
            region: AWS region for Bedrock
            aws_access_key: AWS access key ID (if not using default credentials)
            aws_secret_key: AWS secret access key (if not using default credentials)
            max_chunk_size: Maximum size of code chunks to send to Claude
            max_workers: Maximum number of parallel workers for processing files
            output_dir: Directory to save documentation
            ignore_dirs: List of directory names to ignore
        """
        self.folder_path = os.path.abspath(folder_path)
        self.max_chunk_size = max_chunk_size
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.ignore_dirs = ignore_dirs or ['.git', 'node_modules', 'venv', 'env']
        
        # Initialize Bedrock client
        try:
            self.client = BedrockClient(
                model_id=model_id,
                region=region,
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                temperature=0.2
            )
            # Test connection with a small request
            self.client.generate_text("Say OK", max_tokens=10)
            logger.info("Successfully connected to Amazon Bedrock")
        except Exception as e:
            raise ValueError(f"Failed to initialize Bedrock client: {str(e)}\nPlease check your AWS credentials and permissions.")
        
        # Initialize collections for analysis
        self.all_files = []
        self.programs = {}  # program_id -> file_path
        self.copybooks = {}  # copybook_name -> file_path
        self.program_dependencies = defaultdict(set)  # program -> set of called programs
        self.copybook_usages = defaultdict(set)  # copybook -> set of programs using it
        self.program_descriptions = {}  # program_id -> description
        self.file_types = {}  # file_path -> file_type (PROGRAM, COPYBOOK, JCL, CICS, DB2)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_documentation(self):
        """Main method to orchestrate the documentation generation process."""
        logger.info(f"Starting COBOL documentation generation for: {self.folder_path}")
        
        # Phase 1: Scan files and collect basic information
        self.scan_cobol_files()
        
        # Phase 2: First pass analysis (dependency mapping)
        self.first_pass_analysis()

        # Phase 3: Detailed documentation generation with Claude
        self.second_pass_generate_docs()
        
        # Phase 4: System overview generation
        self.generate_system_overview()
        
        logger.info(f"Documentation generation completed. Output in {self.output_dir}")
    
    def scan_cobol_files(self):
        """Scan the folder for COBOL files and organize them by type."""
        logger.info("Scanning for COBOL files...")
        
        for root, dirs, files in os.walk(self.folder_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for file in files:
                # Include .txt files along with traditional COBOL extensions
                if file.lower().endswith(('.cbl', '.cob', '.cpy', '.pli', '.jcl', '.txt', '.cics', '.sqb')):
                    file_path = os.path.join(root, file)
                    
                    # Determine file type
                    file_type = self.determine_file_type(file_path)
                    self.file_types[file_path] = file_type
                    
                    # For .txt files, check if it contains COBOL code
                    if file.lower().endswith('.txt'):
                        # Simple validation to check if it's likely COBOL code
                        if self.is_likely_cobol(file_path):
                            self.all_files.append(file_path)
                            # Treat text files as programs by default, not copybooks
                            program_name = os.path.splitext(file)[0].upper()
                            if self.extract_program_id_from_file(file_path):
                                program_name = self.extract_program_id_from_file(file_path)
                            self.programs[program_name] = file_path
                    else:
                        self.all_files.append(file_path)
                        
                        # Track copybooks separately
                        if file.lower().endswith('.cpy'):
                            copybook_name = os.path.splitext(file)[0].upper()
                            self.copybooks[copybook_name] = file_path
                        elif file.lower().endswith('.jcl'):  # Track JCL files
                            job_name = self.extract_jcl_job_name(file_path) or os.path.splitext(file)[0].upper()
                            self.programs[job_name] = file_path
                        else:  # Assume non-copybook, non-JCL files are programs
                            # Try to extract program ID from file content
                            program_id = self.extract_program_id_from_file(file_path)
                            if program_id:
                                self.programs[program_id] = file_path
                            else:
                                # Use filename as program ID fallback
                                program_name = os.path.splitext(file)[0].upper()
                                self.programs[program_name] = file_path
        
        if not self.all_files:
            logger.warning(f"No COBOL files found in {self.folder_path}. Ensure the directory contains .cbl, .cob, .cpy, .jcl, or .txt files with COBOL code.")
        else:
            logger.info(f"Found {len(self.all_files)} COBOL files")
            logger.info(f"Identified {len(self.copybooks)} copybooks and {len(self.programs)} potential programs")

    def determine_file_type(self, file_path: str) -> str:
        """
        Determine the type of COBOL file based on extension and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: File type (PROGRAM, COPYBOOK, JCL, CICS, DB2)
        """
        filename = os.path.basename(file_path).lower()
        
        # Check by extension first
        if filename.endswith('.cpy'):
            return 'COPYBOOK'
        elif filename.endswith('.jcl'):
            return 'JCL'
        elif filename.endswith('.cics'):
            return 'CICS'
        elif filename.endswith('.sqb'):
            return 'DB2'
        
        # Check content if not determined by extension
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB
                
                # Check for CICS commands
                if re.search(r'EXEC\s+CICS', content, re.IGNORECASE):
                    return 'CICS'
                
                # Check for SQL statements
                if re.search(r'EXEC\s+SQL', content, re.IGNORECASE):
                    return 'DB2'
                
                # Check for JCL content
                if re.search(r'//\w+\s+JOB', content):
                    return 'JCL'
                
                # Check if it's a copybook (no PROCEDURE DIVISION)
                if not re.search(r'PROCEDURE\s+DIVISION', content, re.IGNORECASE) and not re.search(r'PROGRAM-ID', content, re.IGNORECASE):
                    return 'COPYBOOK'
                    
            # Default to program
            return 'PROGRAM'
            
        except Exception:
            # Default to program if we can't determine
            return 'PROGRAM'

    def is_likely_cobol(self, file_path: str) -> bool:
        """
        Check if a text file likely contains COBOL code by looking for common COBOL keywords.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            bool: True if the file likely contains COBOL code
        """
        try:
            # Read the first 100 lines or fewer if the file is smaller
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = ''.join([next(f) for _ in range(100)])
            
            # Look for common COBOL division headers and statements
            cobol_patterns = [
                r'IDENTIFICATION\s+DIVISION',
                r'DATA\s+DIVISION',
                r'PROCEDURE\s+DIVISION',
                r'PROGRAM-ID',
                r'WORKING-STORAGE\s+SECTION',
                r'FILE\s+SECTION',
                r'LINKAGE\s+SECTION',
                r'MOVE\s+.*\s+TO\s+',
                r'PERFORM\s+',
                r'COMPUTE\s+',
                r'IF\s+.*\s+THEN',
                r'PIC\s+',
                r'DISPLAY\s+',
                r'CALL\s+'
            ]
            
            # Check if any patterns match
            for pattern in cobol_patterns:
                if re.search(pattern, first_lines, re.IGNORECASE):
                    return True
            
            return False
        except Exception:
            return False
        
    def embed_copybooks(self, content: str) -> str:
        """
        Embed copybooks into the COBOL program by replacing COPY statements with the actual content.
        
        Args:
            content: The COBOL source code
            
        Returns:
            str: COBOL source code with embedded copybooks
        """
        copy_pattern = re.compile(r'COPY\s+([A-Za-z0-9-_]+)\s*\.?', re.IGNORECASE)
        matches = list(copy_pattern.finditer(content))
        
        for match in matches:
            copybook_name = match.group(1).upper()
            if copybook_name in self.copybooks:
                copybook_path = self.copybooks[copybook_name]
                try:
                    with open(copybook_path, 'r', encoding='utf-8', errors='ignore') as f:
                        copybook_content = f.read()
                    # Replace the COPY statement with the copybook content
                    content = content.replace(match.group(0), f"* Embedded Copybook: {copybook_name} *\n{copybook_content}")
                    # Log successful embedding
                    logger.info(f"Successfully embedded copybook: {copybook_name}")
                except Exception as e:
                    logger.error(f"Failed to embed copybook {copybook_name}: {str(e)}")
            else:
                logger.warning(f"Copybook {copybook_name} not found. Leaving COPY statement as is.")
        
        return content
    

    def extract_program_id_from_file(self, file_path: str) -> Optional[str]:
        """
        Extract program ID from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Optional[str]: Program ID if found, None otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB to find PROGRAM-ID
                return self.extract_program_id(content)
        except Exception:
            return None
    
    def extract_jcl_job_name(self, file_path: str) -> Optional[str]:
        """
        Extract job name from a JCL file.
        
        Args:
            file_path: Path to the JCL file
            
        Returns:
            Optional[str]: Job name if found, None otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Read first 5KB to find job name
                job_match = re.search(r'//(\w+)\s+JOB', content)
                if job_match:
                    return job_match.group(1).upper()
            return None
        except Exception:
            return None
            
    def first_pass_analysis(self):
        """
        First pass analysis to collect dependency information and basic metadata 
        for all programs in the codebase.
        """
        logger.info("Starting first pass analysis (dependency mapping)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.analyze_file_dependencies, file_path): 
                      file_path for file_path in self.all_files}
            
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                try:
                    file_name = os.path.basename(file_path)
                    future.result()
                    logger.debug(f"Completed dependency analysis for {file_name}")
                except Exception as e:
                    logger.error(f"Error analyzing {file_name}: {str(e)}")
        
        logger.info(f"First pass completed. Found {len(self.programs)} programs with {len(self.program_dependencies)} dependency relationships")
    
    def analyze_file_dependencies(self, file_path: str):
        """Analyze a single file for program ID, COPY statements, and CALL statements."""
        file_type = self.file_types.get(file_path, 'PROGRAM')
        is_copybook = file_type == 'COPYBOOK'
        is_jcl = file_type == 'JCL'
        file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract program ID for program files
            if not is_copybook:
                if is_jcl:
                    program_id = self.extract_jcl_job_name(file_path) or os.path.splitext(file_name)[0].upper()
                    # Find programs executed by JCL
                    for executed_program in self.extract_jcl_exec_programs(content):
                        self.program_dependencies[program_id].add(executed_program)
                else:
                    program_id = self.extract_program_id(content) or os.path.splitext(file_name)[0].upper()
                    self.programs[program_id] = file_path
                    
                    # Find CALL statements
                    for called_program in self.extract_call_targets(content):
                        self.program_dependencies[program_id].add(called_program)
            
            # Extract COPY statements for all files
            program_id = self.extract_program_id(content) or os.path.splitext(file_name)[0].upper()
            for copybook in self.extract_copybooks(content):
                if not is_copybook:  # Only track copybook usage from programs, not from other copybooks
                    self.copybook_usages[copybook].add(program_id)
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    def extract_jcl_exec_programs(self, content: str) -> Set[str]:
        """
        Extract programs executed by JCL.
        
        Args:
            content: JCL content
            
        Returns:
            Set[str]: Set of executed program names
        """
        executed_programs = set()
        
        # Find EXEC PGM= statements
        pgm_matches = re.finditer(r'EXEC\s+PGM=(\w+)', content, re.IGNORECASE)
        for match in pgm_matches:
            executed_programs.add(match.group(1).upper())
        
        # Find EXEC PROC= statements
        proc_matches = re.finditer(r'EXEC\s+(\w+)', content, re.IGNORECASE)
        for match in proc_matches:
            if match.group(1).upper() not in ('PGM'):
                executed_programs.add(match.group(1).upper())
        
        return executed_programs
    
    def second_pass_generate_docs(self):
        """
        Second pass to generate detailed documentation for each file using Claude.
        This involves understanding the business logic and data structures.
        """
        logger.info("Starting second pass (AI-powered documentation generation)...")
        
        # Process files in parallel with a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.document_single_file, file_path): 
                      file_path for file_path in self.all_files}
                
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                file_name = os.path.basename(file_path)
                try:
                    result = future.result()
                    logger.info(f"Documentation completed for {file_name}")
                except Exception as e:
                    logger.error(f"Error documenting {file_name}: {str(e)}")
        
        logger.info("Second pass documentation generation completed")
    
    def document_single_file(self, file_path: str) -> bool:
        """
        Generate comprehensive documentation for a single COBOL file using Claude.
        
        Args:
            file_path: Path to the COBOL file
            
        Returns:
            bool: True if documentation was successful
        """
        file_name = os.path.basename(file_path)
        file_type = self.file_types.get(file_path, 'PROGRAM')
        is_copybook = file_type == 'COPYBOOK'
        is_jcl = file_type == 'JCL'
        is_cics = file_type == 'CICS'
        is_db2 = file_type == 'DB2'
        
        logger.info(f"Generating documentation for {file_name} ({file_type})")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Embed copybooks into the program
            if not is_copybook:  # Only embed copybooks for program files
                original_content = content
                content = self.embed_copybooks(content)
                
                # Save the content only if it was modified
                if content != original_content:
                    # Define the filename
                    filename = f"inserted_copybooks_content_{file_name}.txt"

                    # Write the content to a .txt file
                    with open(filename, "w", encoding="utf-8") as file:
                        file.write(content)

                    logger.info(f"Content successfully saved to {filename}")
            
            # Extract program ID or job name
            if is_jcl:
                program_id = self.extract_jcl_job_name(content) or os.path.splitext(file_name)[0].upper()
            else:
                program_id = self.extract_program_id(content) or os.path.splitext(file_name)[0].upper()
            
            # Prepare additional context about dependencies
            dependency_context = self.prepare_dependency_context(file_path, program_id, file_type)
            
            # Determine if file is large enough to need chunking
            is_large_file = len(content) > self.max_chunk_size * 0.8
            
            # Split content into manageable chunks if needed
            chunks = self.smart_chunking(content) if is_large_file else [content]
            
            # Generate comprehensive documentation using Claude
            documentation = self.generate_documentation_with_claude(
                file_path, program_id, chunks, dependency_context, file_type, is_large_file
            )
            
            # Save the documentation
            rel_path = os.path.relpath(file_path, self.folder_path)
            output_path = os.path.join(self.output_dir, f"{os.path.splitext(rel_path)[0]}.md")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(documentation)
            
            # Store a brief description for system overview
            first_paragraph = documentation.split('\n\n')[1] if len(documentation.split('\n\n')) > 1 else ""
            self.program_descriptions[program_id] = first_paragraph
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to document {file_name}: {str(e)}")
            return False
        
    
    def prepare_dependency_context(self, file_path: str, program_id: str, file_type: str) -> str:
        """
        Prepare context information about program dependencies for Claude.
        
        Args:
            file_path: Path to the COBOL file
            program_id: The program ID
            file_type: Type of the file (PROGRAM, COPYBOOK, JCL, etc.)
            
        Returns:
            str: Context information about dependencies
        """
        context = []
        
        if file_type == 'COPYBOOK':
            # Programs that use this copybook
            copybook_name = os.path.splitext(os.path.basename(file_path))[0].upper()
            if copybook_name in self.copybook_usages:
                using_programs = sorted(self.copybook_usages[copybook_name])
                if using_programs:
                    context.append(f"This copybook is used by: {', '.join(using_programs)}")
        elif file_type == 'JCL':
            # Programs executed by this JCL
            if program_id in self.program_dependencies:
                executed_programs = sorted(self.program_dependencies[program_id])
                if executed_programs:
                    context.append(f"This JCL executes the following programs: {', '.join(executed_programs)}")
        else:
            # Programs called by this program
            if program_id in self.program_dependencies:
                called_programs = sorted(self.program_dependencies[program_id])
                if called_programs:
                    context.append(f"This program calls the following other programs: {', '.join(called_programs)}")
            
            # Programs that call this program
            calling_programs = []
            for caller, called in self.program_dependencies.items():
                if program_id in called:
                    calling_programs.append(caller)
            
            if calling_programs:
                context.append(f"This program is called by: {', '.join(sorted(calling_programs))}")
                
            # Copybooks used by this program
            copybooks_used = []
            for copybook, programs in self.copybook_usages.items():
                if program_id in programs:
                    copybooks_used.append(copybook)
            
            if copybooks_used:
                context.append(f"This program uses these copybooks: {', '.join(sorted(copybooks_used))}")
        
        return "\n".join(context)
    
    def smart_chunking(self, content: str) -> List[str]:
        """
        Split COBOL source into logical chunks while respecting COBOL's structure.
        
        This chunking strategy tries to split at division boundaries first,
        then at section boundaries, and finally at procedure boundaries.
        
        Args:
            content: The COBOL source code
            
        Returns:
            List[str]: List of logical chunks
        """
        # If content is small enough, no need to chunk
        if len(content) <= self.max_chunk_size:
            return [content]
        
        chunks = []
        
        # Try to split at division boundaries
        divisions_pattern = r'(IDENTIFICATION DIVISION|ENVIRONMENT DIVISION|DATA DIVISION|PROCEDURE DIVISION)'
        division_matches = list(re.finditer(divisions_pattern, content, re.IGNORECASE))
        
        if len(division_matches) > 1:
            # We have multiple divisions, split by them
            for i in range(len(division_matches)):
                start = division_matches[i].start()
                end = division_matches[i+1].start() if i < len(division_matches)-1 else len(content)
                division_chunk = content[start:end]
                
                # If this division is too large, we'll need to split it further
                if len(division_chunk) > self.max_chunk_size:
                    # For DATA DIVISION, split by sections and then by entries
                    if "DATA DIVISION" in division_chunk:
                        section_chunks = self.chunk_data_division(division_chunk)
                        chunks.extend(section_chunks)
                    # For PROCEDURE DIVISION, split by sections and paragraphs
                    elif "PROCEDURE DIVISION" in division_chunk:
                        procedure_chunks = self.chunk_procedure_division(division_chunk)
                        chunks.extend(procedure_chunks)
                    else:
                        # For other divisions, just add as is (they're usually smaller)
                        chunks.append(division_chunk)
                else:
                    chunks.append(division_chunk)
        else:
            # Check if this is JCL and split by job steps
            if '//STEP' in content or '// STEP' in content:
                jcl_chunks = self.chunk_jcl(content)
                return jcl_chunks
            
            # Fallback to simple chunking if we don't have clear division markers
            current_chunk = ""
            for line in content.split('\n'):
                if len(current_chunk) + len(line) + 1 > self.max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = line
                else:
                    current_chunk += (line + '\n')
            
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks
    
    def chunk_jcl(self, content: str) -> List[str]:
        """Split JCL into manageable chunks by job steps."""
        chunks = []

        # Get the JOB card and initial comments
        job_start_match = re.search(r'(^.*?JOB\s.*?)(?=//\w+\s+EXEC|\Z)', content, re.DOTALL | re.MULTILINE)
        job_header = job_start_match.group(1) if job_start_match else ""
        
        # Find all EXEC statements which indicate step boundaries
        step_matches = list(re.finditer(r'//(?:\w+)\s+EXEC', content, re.MULTILINE))
        
        if not step_matches:
            # No clear steps, return as is
            return [content]
        
        # Add job header to first chunk
        current_chunk = job_header
        
        for i in range(len(step_matches)):
            start = step_matches[i].start()
            end = step_matches[i+1].start() if i < len(step_matches)-1 else len(content)
            
            step_content = content[start:end]
            
            # If adding this step would make the chunk too large, start a new one
            if len(current_chunk) + len(step_content) > self.max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = job_header + step_content  # Repeat job header for context
            else:
                current_chunk += step_content
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def chunk_data_division(self, content: str) -> List[str]:
        """Split DATA DIVISION into manageable chunks by sections."""
        chunks = []
        
        # Try to split by sections
        sections_pattern = r'(WORKING-STORAGE SECTION|FILE SECTION|LINKAGE SECTION)'
        section_matches = list(re.finditer(sections_pattern, content, re.IGNORECASE))
        
        if not section_matches:
            return [content]
        
        for i in range(len(section_matches)):
            start = section_matches[i].start()
            end = section_matches[i+1].start() if i < len(section_matches)-1 else len(content)
            section_chunk = content[start:end]
            
            # If this section is too large, split it further by entries (01 level items)
            if len(section_chunk) > self.max_chunk_size:
                entry_chunks = self.chunk_by_data_entries(section_chunk)
                chunks.extend(entry_chunks)
            else:
                chunks.append(section_chunk)
        
        return chunks
    
    def chunk_by_data_entries(self, content: str) -> List[str]:
        """Split a data section by 01 level entries."""
        chunks = []
        current_chunk = ""
        header = content.split('\n', 1)[0] + '\n'  # Keep the section header
        
        # Split by 01 level entries
        for line in content.split('\n'):
            # Start of a new 01 level item
            if re.match(r'^\s*01\s+', line):
                if current_chunk and len(current_chunk) > len(header):
                    chunks.append(current_chunk)
                    current_chunk = header + line + '\n'
                else:
                    current_chunk += line + '\n'
            else:
                current_chunk += line + '\n'
                
                # If chunk gets too large, break it
                if len(current_chunk) > self.max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = header
        
        if current_chunk and len(current_chunk) > len(header):
            chunks.append(current_chunk)
            
        return chunks
    
    def chunk_procedure_division(self, content: str) -> List[str]:
        """Split PROCEDURE DIVISION into manageable chunks by sections or paragraphs."""
        chunks = []
        
        # Try to split by sections first
        section_pattern = r'(^[A-Za-z0-9-_]+\s+SECTION\.)'
        section_matches = list(re.finditer(section_pattern, content, re.MULTILINE))
        
        header = "PROCEDURE DIVISION."
        procedure_header_match = re.search(r'(PROCEDURE\s+DIVISION[^.]*\.)', content, re.IGNORECASE)
        if procedure_header_match:
            header = procedure_header_match.group(1)
        
        if section_matches:
            for i in range(len(section_matches)):
                start = section_matches[i].start()
                end = section_matches[i+1].start() if i < len(section_matches)-1 else len(content)
                section_chunk = content[start:end]
                
                if len(section_chunk) > self.max_chunk_size:
                    # Split this section by paragraphs
                    paragraph_chunks = self.chunk_by_paragraphs(section_chunk)
                    chunks.extend(paragraph_chunks)
                else:
                    chunks.append(section_chunk)
        else:
            # If no sections, split by paragraphs directly
            paragraph_chunks = self.chunk_by_paragraphs(content)
            chunks.extend(paragraph_chunks)
        
        return chunks
    
    def chunk_by_paragraphs(self, content: str) -> List[str]:
        """Split content by paragraph boundaries."""
        chunks = []
        lines = content.split('\n')
        
        # Extract header (first line or PROCEDURE DIVISION statement)
        header = lines[0] + '\n'
        
        current_chunk = header
        current_paragraph = None
        
        for line in lines[1:]:  # Skip header which we already added
            # Check for paragraph start (non-indented name followed by period)
            paragraph_match = re.match(r'^([A-Za-z0-9-_]+)\.\s*$', line)
            
            if paragraph_match:
                # If we have accumulated enough content, start a new chunk
                if len(current_chunk) > self.max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = header + line + '\n'
                else:
                    current_chunk += line + '\n'
                
                current_paragraph = paragraph_match.group(1)
            else:
                current_chunk += line + '\n'
                
                # If chunk gets too large and we're not in the middle of a statement
                if len(current_chunk) > self.max_chunk_size and '.' in line:
                    chunks.append(current_chunk)
                    current_chunk = header
        
        if len(current_chunk) > len(header):
            chunks.append(current_chunk)
            
        return chunks
    
    def generate_documentation_with_claude(self, file_path: str, program_id: str, 
                                         chunks: List[str], dependency_context: str,
                                         file_type: str, is_large_file: bool) -> str:
        """
        Generate comprehensive documentation using Claude AI by processing chunks
        and merging the results.
        
        Args:
            file_path: Path to the COBOL file
            program_id: The program ID
            chunks: List of code chunks
            dependency_context: Context about dependencies
            file_type: Type of the file (PROGRAM, COPYBOOK, JCL, etc.)
            is_large_file: Whether this is a large file
            
        Returns:
            str: Generated documentation
        """
        file_name = os.path.basename(file_path)
        
        if len(chunks) == 1:
            # Single chunk processing
            return self.process_chunk_with_claude(
                chunks[0], file_name, program_id, dependency_context, file_type, 
                is_partial=False
            )
        else:
            # Multi-chunk processing with a two-phase approach
            partial_docs = []

            # Phase 1: Process each chunk for partial documentation
            logger.info(f"Processing {len(chunks)} chunks for {file_name}")
            
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)} for {file_name}")
                partial_doc = self.process_chunk_with_claude(
                    chunk, file_name, program_id, dependency_context, file_type, 
                    is_partial=True, chunk_num=i+1, total_chunks=len(chunks)
                )
                partial_docs.append(partial_doc)
                # After populating partial_docs
                partial_docs_file_path = os.path.join(self.output_dir, f"{file_name}_partial_docs.md")
                try:
                    with open(partial_docs_file_path, 'w', encoding='utf-8') as partial_docs_file:
                        partial_docs_file.write("# Partial Documentation\n\n")
                        for i, partial_doc in enumerate(partial_docs, start=1):
                            partial_docs_file.write(f"## Chunk {i}\n\n")
                            partial_docs_file.write(partial_doc)
                            partial_docs_file.write("\n\n")
                    logger.info(f"Partial documentation saved to {partial_docs_file_path}")
                except Exception as e:
                    logger.error(f"Failed to save partial documentation for {file_name}: {str(e)}")
                time.sleep(1)  # Small delay to avoid rate limiting
            
            # Phase 2: Consolidate the partial documents
            logger.debug(f"Consolidating documentation for {file_name}")
            consolidated_prompt = f"""
You are a COBOL expert who will do the rule harvesting and documentation generation for a COBOL program.

I'll provide you with partial documentation segments generated from different chunks of the {file_type} '{file_name}' with ID '{program_id}'.

Your task is to consolidate these segments into a single, coherent, comprehensive documentation that includes all conditional statements especifically available under data division, procedure, paragraph level in plain English.
1. Program/File Overview
2. Functional Analysis
3. Technical Structure
4. Data Structures
5. Processing Logic
6. I/O Operations
7. Error Handling and Validation
8. Integration Points
9. Performance Considerations
10. Technical Debt and Modernization Opportunities
11. Code Rules and Logic

Resolve any redundancies, organize the information logically, and ensure all important aspects are covered.

The final documentation should be in Markdown format with clear headings and sections.

Here are the partial documentation segments to consolidate:

{'-' * 40}
{"".join(partial_docs)}
{'-' * 40}

Create a comprehensive, consolidated documentation that explains both business functionality and technical implementation.
"""
            
            # Get consolidated documentation from Claude
            try:
                consolidated_doc = self.client.generate_text(
                    prompt=consolidated_prompt,
                    system_prompt=self.get_system_prompt_by_file_type(file_type, is_large_file),
                    max_tokens=100000
                )
                return consolidated_doc
            except Exception as e:
                logger.error(f"Error consolidating documentation for {file_name}: {str(e)}")
                # Fall back to joining the partial docs if consolidation fails
                return f"# {file_type}: {program_id} ({file_name})\n\n" + "\n\n".join(partial_docs)
    
    def get_system_prompt_by_file_type(self, file_type: str, is_large_file: bool) -> str:
        """
        Get the appropriate system prompt based on file type.
        
        Args:
            file_type: Type of file (PROGRAM, COPYBOOK, JCL, CICS, DB2)
            is_large_file: Whether the file is large
            
        Returns:
            str: System prompt for Claude
        """
        base_prompt = "You are a COBOL expert who creates detailed, comprehensive documentation from COBOL source code. Your documentation explains business logic, calculations, data structures, and program flow in a clear, accessible way for both technical and business audiences."
        
        if is_large_file:
            base_prompt += " For large programs, focus on the most important aspects: overall architecture, main business functions, critical sections, integration points, and core data structures."
        
        if file_type == 'COPYBOOK':
            return base_prompt + " For COPYBOOK analysis, focus especially on the data structures defined, their business meaning, how they're used by programs, any business rules embedded in the data definitions, and field-by-field explanations."
        elif file_type == 'JCL':
            return base_prompt + " For JCL analysis, focus especially on the job workflow and execution sequence, program parameters, file allocations, job scheduling and dependencies, and condition code processing."
        elif file_type == 'CICS':
            return base_prompt + " For CICS programs, explain transaction flow and screen interactions, CICS command usage, terminal I/O operations, state management, and CICS-specific error handling."
        elif file_type == 'DB2':
            return base_prompt + " For programs with DB2 operations, explain database operations, table structures and their business meaning, join logic, transaction management, and SQL optimization considerations."
        else:
            return base_prompt
        
    def process_chunk_with_claude(self, chunk: str, file_name: str, program_id: str,
                               dependency_context: str, file_type: str,
                               is_partial: bool = False, chunk_num: int = 1, 
                               total_chunks: int = 1) -> str:
        """
        Process a single chunk of COBOL code with Claude to generate documentation.
        
        Args:
            chunk: The COBOL code chunk
            file_name: Name of the file
            program_id: The program ID
            dependency_context: Context about dependencies
            file_type: Type of file (PROGRAM/COPYBOOK/JCL/CICS/DB2)
            is_partial: Whether this is a partial chunk of a larger file
            chunk_num: Current chunk number
            total_chunks: Total number of chunks
            
        Returns:
            str: Generated documentation
        """
        chunk_desc = f" (Part {chunk_num} of {total_chunks})" if is_partial else ""

        # Save the chunk to a text file for debugging or review
        chunk_output_dir = os.path.join(self.output_dir, "chunks")
        os.makedirs(chunk_output_dir, exist_ok=True)
        chunk_file_path = os.path.join(chunk_output_dir, f"{file_name}_chunk_{chunk_num}.txt")
        with open(chunk_file_path, 'w', encoding='utf-8') as chunk_file:
            chunk_file.write(chunk)
        
        
        # Base prompt with comprehensive analysis requirements
        base_prompt = f"""
# COBOL Rule Harvesting Request

## Source Information
{'Program' if file_type != 'COPYBOOK' else 'Copybook'} Name: {program_id}
File Type: {file_type}
Filename: {file_name}{chunk_desc}

Dependencies information:
{dependency_context}

SOURCE CODE:
```cobol
{chunk}
```

## Analysis Requirements

Please provide a rule harvesting of this COBOL component with the following sections:

1. {'Program' if file_type != 'COPYBOOK' else 'Copybook'} Overview
   - Provide a concise summary of the primary business purpose
   - Identify the type {'of program (batch, online, etc.)' if file_type != 'COPYBOOK' else 'of data structures defined'}
     
2. Functional Analysis
   - Describe the complete and detailed business rules and logic implemented
   - Describe the expected business outcomes when {'this program executes' if file_type != 'COPYBOOK' else 'these structures are used'}
   
3. Data Structures
   {'- Document all key data structures (FD, 01 level) with their business purpose' if file_type != 'COPYBOOK' else '- Document all data structures with their business purpose'}
   {'- Explain file record layouts and their field definitions' if file_type != 'COPYBOOK' else '- Explain all field definitions in detail'}
   {'- Describe WORKING-STORAGE variables and their purposes' if file_type != 'COPYBOOK' else ''}
   - Identify key data transformations and mappings
   - Document any complex data manipulations
   - Create a COBOL file structure for input and output file. The file should use fixed-length records and contain fields for identifying the file attributes and 
     format .Additionally, include data types and data size for ensuring proper formatting with signed values and implied decimals where necessary. Use filler spaces strategically to maintain record alignment.
     The final record should feature a message code and a descriptive name field.

4. {'Processing Logic' if file_type != 'COPYBOOK' else 'Usage Patterns'}
   {'- Provide a flowchart-like explanation of the main processing steps' if file_type != 'COPYBOOK' else '- Provide patterns for how programs typically use this copybook'}
   {'- Document the conditional logic and decision points without adding cobol' if file_type != 'COPYBOOK' else ''}
   {'- Explain any complex algorithms or calculations' if file_type != 'COPYBOOK' else '- Explain any business rules embedded in the data definitions'}
   
5. User Story Creation
   - Follow the steps outlined in the 'User Story Creation' section to generate detailed user stories for this COBOL component. Ensure that user stories are created from the perspective of every relevant persona, accurately capturing their unique needs and interactions.
   - Ensure the user stories are actionable, aligned with the program's functionality, and include clear acceptance criteria and success metrics.

## User Story Creation

Using the provided COBOL code, create detailed user stories that align with the program's functionality and user needs. Follow these steps:

1. **Identify User Needs**:
   - What problem does the COBOL program solve for the users?
   - What are the key features that users rely on in the current program?
   - Are there any pain points or limitations in the current COBOL program that users have expressed?

2. **Define the User Story**:
   - Use the format: **As a [type of user], I want to [perform a specific task] so that I can [achieve a specific goal].**
   - Clearly define the acceptance criteria for the user story.
   - Specify success metrics to measure the effectiveness of the user story.

3. **Map COBOL Functions to User Stories**:
   - Analyze the COBOL code to identify functions or operations that directly impact users.
   - Look for input/output operations, reports, or user interactions in the code.
   - Identify business rules or logic that address specific user needs or solve user problems. Don't add any cobol code directly
   - Map these functions to user stories by determining how they align with user tasks and goals.

4. **Create Detailed User Stories**:
   - Break down the main user story into smaller, actionable user stories.
   - Prioritize these user stories based on user needs and business value.
   - Ensure each user story has clear acceptance criteria and success metrics.

5. **Extract User-Centric Information**:
   - Identify any user-facing operations in the COBOL code (e.g., `DISPLAY`, `ACCEPT`, `PRINT` statements).
   - Look for business rules or logic that directly address user needs or solve user problems.
   - Highlight any dependencies or workflows that impact user tasks.

### Example User Stories

**Example 1: Automate Payroll Processing**
- **As a** payroll administrator, **I want to** automate the payroll processing system **so that** I can reduce manual errors and save time on payroll calculations.
- **Acceptance Criteria**:
  1. The system should automatically calculate employee salaries based on their hours worked and pay rate.
  2. The system should apply deductions such as taxes, insurance, and other withholdings accurately.
  3. The system should generate payroll reports that can be reviewed and approved by the payroll administrator.
  4. The system should allow for adjustments to be made in case of discrepancies.
  5. The system should be able to handle different pay periods (weekly, bi-weekly, monthly).
- **Success Metrics**:
  - Reduction in payroll processing time by 50%.
  - Decrease in payroll errors by 90%.
  - Positive feedback from payroll administrators on the ease of use and accuracy of the system.

**Example 2: Enhance Inventory Management System**
- **As a** warehouse manager, **I want to** improve the inventory management system **so that** I can track stock levels more accurately and streamline the restocking process.
- **Acceptance Criteria**:
  1. The system should provide real-time updates on stock levels as items are added or removed from inventory.
  2. The system should generate automatic restocking alerts when stock levels fall below predefined thresholds.
  3. The system should integrate with the company's ERP system to ensure consistency across all business operations.
  4. The system should allow warehouse managers to view inventory levels, restocking alerts, and historical data through a user-friendly interface.
  5. The system should support barcode scanning for efficient inventory management.
- **Success Metrics**:
  - Reduction in stock discrepancies by 80%.
  - Decrease in restocking delays by 70%.
  - Positive feedback from warehouse managers on the accuracy and usability of the system.

**Example 3: Generate Monthly Sales Report** 
    IDENTIFICATION DIVISION.
    PROGRAM-ID. SALES-REPORT.
    DATA DIVISION.
    WORKING-STORAGE SECTION.
    01 WS-TOTAL-SALES PIC 9(9) VALUE ZERO.
    PROCEDURE DIVISION.
    DISPLAY "ENTER MONTH: " ACCEPT WS-MONTH.
    PERFORM CALCULATE-SALES.
    DISPLAY "TOTAL SALES FOR " WS-MONTH " IS " WS-TOTAL-SALES.
    STOP RUN.
**As a** sales manager, **I want to** generate a monthly sales report **so that** I can track sales performance and identify trends.
**Acceptance Criteria**:
1. The system should prompt the user to enter a month.
2. The system should calculate total sales for the specified month.
3. The system should display the total sales in a user-friendly format.
**Success Metrics**:
- Reduction in time taken to generate sales reports by 80%.
- Positive feedback from sales managers on the accuracy and usability of the report.

**Example 4: Process Employee Payroll**
    IDENTIFICATION DIVISION.
    PROGRAM-ID. PAYROLL.
    DATA DIVISION.
    WORKING-STORAGE SECTION.
    01 WS-NET-PAY PIC 9(9)V99 VALUE ZERO.
    PROCEDURE DIVISION.
    DISPLAY "ENTER EMPLOYEE ID: " ACCEPT WS-EMP-ID.
    PERFORM CALCULATE-PAY.
    DISPLAY "NET PAY FOR EMPLOYEE " WS-EMP-ID " IS " WS-NET-PAY.
    STOP RUN.
**As a** payroll administrator, **I want to** calculate employee net pay **so that** I can ensure accurate and timely payroll processing.
**Acceptance Criteria**:
1. The system should prompt the user to enter an employee ID.
2. The system should calculate the net pay based on hours worked, deductions, and bonuses.
3. The system should display the net pay in a user-friendly format.
**Success Metrics**:
- Reduction in payroll processing errors by 90%.
- Positive feedback from payroll administrators on the accuracy and usability of the system.
  """

        # Final instruction for all file types
        base_prompt += """
Format your response in clear Markdown with appropriate headings, code references, and explanations targeted at both technical and business audiences.
"""

        # Special instruction for partial documentation
        if is_partial:
            base_prompt += f"""
Since this is part {chunk_num} of {total_chunks}, focus on documenting what's visible in this specific chunk. The full documentation will be consolidated from all parts later.
"""

        # For large files, add special instructions
        if total_chunks > 3:
            base_prompt += """
This is a large program, so focus on the most important aspects visible in this chunk:
1. Key business functions and logic
2. Critical data structures and their business meaning
3. Main processing flow and important algorithms
4. Significant integration points and dependencies
"""

        chunk_title = f"# {file_type}: {program_id} ({file_name}){chunk_desc}\n\n" if not is_partial else ""
        
        try:
            # Use Claude via Bedrock to generate the documentation
            system_prompt = self.get_system_prompt_by_file_type(file_type, total_chunks > 3)
            
            response = self.client.generate_text(
                prompt=base_prompt,
                system_prompt=system_prompt,
                max_tokens=64000
            )
            
            return chunk_title + response
                
        except Exception as e:
            logger.error(f"Error generating documentation for {file_name} chunk {chunk_num}: {str(e)}")
            return f"{chunk_title}Error generating documentation: {str(e)}"
    
    def generate_system_overview(self):
        """Generate a comprehensive system overview document."""
        logger.info("Generating system overview documentation...")
        
        try:
            overview = f"""# COBOL System Documentation

## System Overview

This documentation was generated on {datetime.now().strftime("%Y-%m-%d %H:%M")} using Claude AI.

### Statistics

- Total COBOL files: {len(self.all_files)}
- Programs: {len([f for f in self.file_types.values() if f == 'PROGRAM' or f == 'CICS' or f == 'DB2'])}
- Copybooks: {len([f for f in self.file_types.values() if f == 'COPYBOOK'])}
- JCL Jobs: {len([f for f in self.file_types.values() if f == 'JCL'])}
- CICS Programs: {len([f for f in self.file_types.values() if f == 'CICS'])}
- DB2/SQL Programs: {len([f for f in self.file_types.values() if f == 'DB2'])}

## Program Index

| Program ID | Type | Description | Dependencies |
|------------|------|-------------|--------------|
"""
            
            # Add program information
            for program_id, file_path in sorted(self.programs.items()):
                file_type = self.file_types.get(file_path, 'PROGRAM')
                if file_type == 'COPYBOOK':
                    continue  # Skip copybooks for this section
                
                file_name = os.path.basename(file_path)
                rel_path = os.path.relpath(file_path, self.folder_path)
                doc_link = f"[{file_name}]({os.path.splitext(rel_path)[0]}.md)"
                
                description = self.program_descriptions.get(program_id, "")
                if len(description) > 100:
                    description = description[:97] + "..."
                
                # Get dependencies
                dependencies = list(self.program_dependencies.get(program_id, set()))
                dependency_str = ", ".join(dependencies) if dependencies else "None"
                
                overview += f"| {program_id} | {file_type} | {description} | {dependency_str} |\n"
            
            # Add copybook information
            overview += "\n## Copybook Index\n\n"
            overview += "| Copybook | Used By |\n"
            overview += "|----------|--------|\n"
            
            for copybook, file_path in sorted(self.copybooks.items()):
                file_name = os.path.basename(file_path)
                rel_path = os.path.relpath(file_path, self.folder_path)
                doc_link = f"[{file_name}]({os.path.splitext(rel_path)[0]}.md)"
                
                # Get programs using this copybook
                using_programs = sorted(self.copybook_usages.get(copybook, set()))
                programs_str = ", ".join(using_programs) if using_programs else "Not referenced"
                
                overview += f"| {copybook} | {programs_str} |\n"
            
            # Generate program dependency graph
            overview += "\n## Program Dependencies\n\n"
            overview += "The following sections outline the call hierarchy and dependencies between programs.\n\n"
            
            # List programs by their calling relationships
            for program_id, called_programs in sorted(self.program_dependencies.items()):
                if called_programs:
                    overview += f"### {program_id}\n\n"
                    overview += "Calls the following programs:\n\n"
                    for called in sorted(called_programs):
                        if called in self.programs:
                            rel_path = os.path.relpath(self.programs[called], self.folder_path)
                            doc_link = f"[{called}]({os.path.splitext(rel_path)[0]}.md)"
                            overview += f"- {doc_link}\n"
                        else:
                            overview += f"- {called} _(external)_\n"
                    overview += "\n"
            
            # Add system architecture visualization section
            overview += "\n## System Architecture\n\n"
            overview += "The system consists of the following major components and their interactions:\n\n"
            
            # Group programs by type
            program_types = {}
            for file_path, file_type in self.file_types.items():
                if file_path in self.programs.values():  # Only look at programs, not copybooks
                    program_id = next(pid for pid, path in self.programs.items() if path == file_path)
                    if file_type not in program_types:
                        program_types[file_type] = []
                    program_types[file_type].append(program_id)
            
            # Describe each program type group
            for file_type, programs in program_types.items():
                if programs:
                    overview += f"### {file_type} Components\n\n"
                    overview += f"The system includes {len(programs)} {file_type.lower()} components:\n\n"
                    for program_id in sorted(programs)[:10]:  # List up to 10 for brevity
                        description = self.program_descriptions.get(program_id, "")
                        short_desc = description[:100] + "..." if len(description) > 100 else description
                        overview += f"- **{program_id}**: {short_desc}\n"
                    
                    if len(programs) > 10:
                        overview += f"- _(plus {len(programs) - 10} more...)_\n"
                    
                    overview += "\n"
            
            # Save the system overview
            with open(os.path.join(self.output_dir, "system_overview.md"), 'w', encoding='utf-8') as f:
                f.write(overview)
                
            logger.info("System overview documentation completed")
            
        except Exception as e:
            logger.error(f"Error generating system overview: {str(e)}")
    
    def extract_program_id(self, content: str) -> Optional[str]:
        """
        Extract the PROGRAM-ID from COBOL code.
        
        Args:
            content: The COBOL source code
            
        Returns:
            Optional[str]: The program ID if found, otherwise None
        """
        # Look for standard PROGRAM-ID declaration
        match = re.search(r'PROGRAM-ID\s*\.\s*([A-Za-z0-9-_]+)', content, re.IGNORECASE)
        if match:
            return match.group(1).strip().upper()
        return None
    
    def extract_call_targets(self, content: str) -> Set[str]:
        """
        Extract programs called via CALL statements.
        
        Args:
            content: The COBOL source code
            
        Returns:
            Set[str]: Set of called program names
        """
        called_programs = set()
        
        # Match literal string calls: CALL 'PROGRAM'
        literal_calls = re.finditer(r'CALL\s+[\'"]([A-Za-z0-9-_]+)[\'"]', content, re.IGNORECASE)
        for match in literal_calls:
            called_programs.add(match.group(1).upper())
        
        # Match direct identifier calls: CALL PROGRAM
        direct_calls = re.finditer(r'CALL\s+(?![\'"$])([\w-]+)', content, re.IGNORECASE)
        for match in direct_calls:
            # Only consider if it's likely a literal program name, not a variable
            program_name = match.group(1).upper()
            if not any(x in program_name for x in ['WS-', 'PGM-', 'PROGRAM-']):
                called_programs.add(program_name)
                
        # For CICS programs, also check for LINK and XCTL commands
        cics_calls = re.finditer(r'EXEC\s+CICS\s+(LINK|XCTL)\s+PROGRAM\s*\(\s*[\'"]?([A-Za-z0-9-_]+)[\'"]?\s*\)', content, re.IGNORECASE)
        for match in cics_calls:
            called_programs.add(match.group(2).upper())
        
        return called_programs
    
    def extract_copybooks(self, content: str) -> Set[str]:
        """
        Extract copybooks referenced via COPY statements.
        
        Args:
            content: The COBOL source code
            
        Returns:
            Set[str]: Set of copybook names
        """
        copybooks = set()
        
        # Match COPY statements
        copy_matches = re.finditer(r'COPY\s+([A-Za-z0-9-_]+)', content, re.IGNORECASE)
        for match in copy_matches:
            copybooks.add(match.group(1).upper())
        
        return copybooks


def main():
    """Main function to run the COBOL documentation generator."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive documentation from COBOL source files using Claude AI via Amazon Bedrock."
    )
    
    parser.add_argument("--folder_path", help="Path to the folder containing COBOL files")
    parser.add_argument("--aws-access-key", help="AWS access key ID")
    parser.add_argument("--aws-secret-key", help="AWS secret access key")
    parser.add_argument("--profile", help="AWS profile name to use from credentials file")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--model", default="us.anthropic.claude-3-7-sonnet-20250219-v1:0", help="Claude model ID to use in Bedrock")
    parser.add_argument("--output", default="documentation", help="Output directory for documentation")
    parser.add_argument("--max-chunk", type=int, default=80000, help="Maximum chunk size in characters")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Check AWS credentials before initializing the generator
        if args.profile:
            # Use the specified profile
            os.environ['AWS_PROFILE'] = args.profile
            logger.info(f"Using AWS profile: {args.profile}")

            # Verify AWS credentials are accessible
            try:
                session = boto3.Session(
                    aws_access_key_id=args.aws_access_key,
                    aws_secret_access_key=args.aws_secret_key,
                    region_name=args.region
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

        # Initialize and run the documentation generator
        generator = CobolDocumentationGenerator(
            folder_path=args.folder_path,
            model_id=args.model,
            region=args.region,
            aws_access_key=args.aws_access_key,
            aws_secret_key=args.aws_secret_key,
            max_chunk_size=args.max_chunk,
            max_workers=args.max_workers,
            output_dir=args.output
        )
        
        generator.generate_documentation()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
                
        
