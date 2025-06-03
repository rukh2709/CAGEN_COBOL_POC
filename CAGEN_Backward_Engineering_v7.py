import os
import re
import sys
import json
import logging
from datetime import datetime
import argparse

# Logging setup 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cagen_backward_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config
except ImportError:
    logger.error("The 'boto3' package is not installed. Please install it with: pip install boto3")
    sys.exit(1)

class BedrockClient:
    def __init__(self, model_id="anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0.0, 
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

    def generate_text(self, prompt, system_prompt=None, max_tokens=64000):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "messages": messages
            }
            if system_prompt:
                if "anthropic.claude" in self.model_id:
                    request_body["system"] = system_prompt
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            response_body = json.loads(response.get('body').read())
            if 'content' in response_body and len(response_body['content']) > 0:
                output_text = response_body['content'][0]['text']
                return output_text
            else:
                logger.error(f"Unexpected response structure: {response_body}")
                return "Error: Unexpected response format from Claude"
        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}")
            raise e

embedded_cache = {}
called_mrns = set()
call_tree = {}

def read_component_file(component_id):
    filename = f"{component_id[:3]}{component_id[3:8]}.txt"
    filepath = os.path.join(COMPONENT_DIR, filename)
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return f"# [Missing component: {component_id}]"
    with open(filepath, "r") as f:
        return f.read().rstrip()

def add_to_call_tree(parent, child):
    if parent not in call_tree:
        call_tree[parent] = []
    if parent not in call_tree[parent]: 
        call_tree[parent].append(child)

def embed_components(text, parent="ROOT"):
    lines = text.splitlines()
    output = []

    for line in lines:
        match = re.match(r"USE\s+([A-Z]{3}\d{5})", line.strip())
        if match:
            comp_id = match.group(1)
            add_to_call_tree(parent, comp_id)

            # # Add relationship line before embedding
            # output.append(f"# {parent} is calling {comp_id}")

            if parent in called_mrns or comp_id in called_mrns:
                print(f"Found USE: {comp_id} inside {parent} — embedding...")
            else:
                print(f"Found USE: {comp_id} inside {parent}, but {parent} is not directly called in IRN — skipping embed in output.")

            if comp_id in embedded_cache:
                output.append(f"# Embedded: {comp_id} [Skipped - Already Included]")
                continue

            embedded_cache[comp_id] = True
            comp_content = read_component_file(comp_id)

            if comp_id.startswith("IRN"):
                nested_mrns = extract_mrns_from_text(comp_content)
                called_mrns.update(nested_mrns)

            embedded_block = embed_components(comp_content, parent=comp_id)
            output.append(f"# Start of embedded component: {comp_id}\n{embedded_block.strip()}\n# End of embedded component: {comp_id}")
        else:
            output.append(line.rstrip())

    return "\n".join([line for line in output if line.strip()])

def extract_mrns_from_text(text):
    return re.findall(r"USE\s+(MRN\d{5})", text)

def extract_mrns_from_irn(irn_file_path):
    with open(irn_file_path, "r") as f:
        irn_text = f.read()
    mrn_matches = extract_mrns_from_text(irn_text)
    print(f"MRNs found in {irn_file_path}: {mrn_matches}")
    called_mrns.update(mrn_matches)
    return set(mrn_matches)

def log_call_tree(output_path="component_tree.log"):
    def walk_tree(node, prefix=""):
        lines = [f"{prefix}{node}"]
        children = call_tree.get(node, [])
        for i, child in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            lines.extend(walk_tree(child, prefix + "    " if i == len(children) - 1 else prefix + "│   "))
        return lines

    roots = [k for k in call_tree if not any(k in v for v in call_tree.values())]
    all_lines = []
    for root in roots:
        all_lines.extend(walk_tree(root))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"\nComponent tree saved to: {output_path}")

def detect_entry_irns():
    all_irns = set()
    used_irns = set()
   
    for filename in os.listdir(COMPONENT_DIR):
        if filename.startswith("IRN") and filename.endswith(".txt"):
            irn_id = filename.replace(".txt", "")
            all_irns.add(irn_id)
            print(f"Found IRN: {irn_id}")

        # Scan all files for USE IRNxxxxx
        filepath = os.path.join(COMPONENT_DIR, filename)
        if not filename.endswith(".txt"):
            continue
        with open(filepath, "r") as f:
            content = f.read()
            found = re.findall(r"USE\s+(IRN\d{5})", content)
            used_irns.update(found)
    print(f"\n All IRNs: {all_irns}")
    print(f"\n Used IRNs: {used_irns}")
    entry_irns = all_irns - used_irns
    print(f"\n Auto-detected entry IRNs: {entry_irns}")
    return entry_irns

def process_irn(irn_file_name, claude=None, output_dir=None):
    irn_file_path = os.path.join(COMPONENT_DIR, irn_file_name)
    mrn_ids = extract_mrns_from_irn(irn_file_path)

    top_irn = irn_file_name.replace(".txt", "")
    call_tree[top_irn] = set(mrn_ids)
    logger.info(f"Processing IRN: {top_irn} with MRNs: {mrn_ids}")

    for mrn_id in mrn_ids:
        logger.info(f"Processing MRN: {mrn_id}")
        print(f"\nProcessing MRN: {mrn_id}")
        embedded_cache.clear()
        embedded_cache[mrn_id] = True
        mrn_text = read_component_file(mrn_id)
        full_expanded = embed_components(mrn_text, parent=mrn_id)
        # Add header indicating the relationship
        irn_id = top_irn  # top_irn is the IRN file (without .txt)
        header_line = f"{mrn_id} is called by {irn_id}\n"
        # Save expanded MRN to output_dir
        expanded_file = os.path.join(output_dir, f"{mrn_id}_expanded.md")
        with open(expanded_file, "w") as out:
            out.write(header_line)
            out.write(full_expanded.strip() + "\n")
        logger.info(f"Saved expanded MRN to {expanded_file}")
        print(f"Saved expanded MRN to {expanded_file}")

        # Summarize the expanded MRN file using Claude if client is provided
        if claude:
            try:
                logger.info(f"Sending expanded MRN {mrn_id} to Claude for summarization.")
                system_prompt = (
    "You are an expert CAGEN documentation generator. Your task is to analyze CAGEN component source code, "
    "including any embedded or called components, and produce clear, structured, and comprehensive technical documentation. "
    "Explain the SRN component details in plain english."
    "The documentation should explain the purpose, logic, data flows, inputs, outputs, and relationships between all components. "
    "Use headings, bullet points, and code blocks where appropriate. Assume the reader is a technical audience but may not be familiar with the specific application."
)

                user_prompt = (
    "Analyze the following CAGEN component and all its embedded components. For each component, provide:\n"
    "- A summary of its purpose and functionality.\n"
    "- Detailed explanation of its logic and processing flow.\n"
    "- Description of all inputs (imports), outputs (exports), and local variables.\n"
    "- Explanation of how embedded components are used, including their logic and how data flows between them.\n"
    "- Any important business rules, conditions, reason codes, return codes, and error handling present in the code.\n"
    "- A diagram or structured outline of the relationships and call hierarchy between components including the top most IRN that calls the MRN's (if possible).\n\n"
    "Document everything in a clear, organized, and technical manner suitable for developers and system analysts.\n\n"
    "## User Story Creation\n"
    "Using the provided CAGEN code, create detailed user stories(atleast 5) that align with the program's functionality and user needs. Follow these steps:\n\n"
    "1. **Identify User Needs**:\n"
    "   - What problem does the program solve for the users?\n"
    "   - What are the key features that users rely on in the current program?\n"
    "   - Are there any pain points or limitations in the current program that users have expressed?\n\n"
    "2. **Define the User Story**:\n"
    "   - Use the format: **As a [type of user], I want to [perform a specific task] so that I can [achieve a specific goal].**\n"
    "   - Clearly define the acceptance criteria for the user story.\n"
    "   - Specify testing criteria and test scenarios for the user story.\n"
    "   - Specify success metrics to measure the effectiveness of the user story.\n\n"
    "3. **Map Program Functions to User Stories**:\n"
    "   - Analyze the code to identify functions or operations that directly impact users.\n"
    "   - Look for input/output operations, reports, or user interactions in the code.\n"
    "   - Identify business rules or logic that address specific user needs or solve user problems.\n"
    "   - Identify test cases that address specific user needs or solve user problems.\n"
    "   - Map these functions to user stories by determining how they align with user tasks and goals.\n\n"
    "4. **Create Detailed User Stories**:\n"
    "   - Break down the main user story into smaller, actionable user stories.\n"
    "   - Prioritize these user stories based on user needs ,test scenarios and business value.\n"
    "   - Ensure each user story has clear acceptance criteria and success metrics.\n\n"
    "5. **Extract User-Centric Information**:\n"
    "   - Identify any user-facing operations in the code (e.g., DISPLAY, ACCEPT, PRINT statements).\n"
    "   - Look for business rules or logic that directly address user needs or solve user problems.\n"
    "   - Look for test cases that directly address user needs or solve user problems.\n"
    "   - Highlight any dependencies or workflows that impact user tasks.\n\n"
    "### Example User Stories\n"
    "**Example 1: Automate Payroll Processing**\n"
    "- **As a** payroll administrator, **I want to** automate the payroll processing system **so that** I can reduce manual errors and save time on payroll calculations.\n"
    "- **Acceptance Criteria**:\n"
    "  1. The system should automatically calculate employee salaries based on their hours worked and pay rate.\n"
    "  2. The system should apply deductions such as taxes, insurance, and other withholdings accurately.\n"
    "  3. The system should generate payroll reports that can be reviewed and approved by the payroll administrator.\n"
    "  4. The system should allow for adjustments to be made in case of discrepancies.\n"
    "  5. The system should be able to handle different pay periods (weekly, bi-weekly, monthly).\n"
    "- **Success Metrics**:\n"
    "  - Reduction in payroll processing time by 50%.\n"
    "  - Decrease in payroll errors by 90%.\n"
    "  - Positive feedback from payroll administrators on the ease of use and accuracy of the system.\n\n"
    "**Example 2: Enhance Inventory Management System**\n"
    "- **As a** warehouse manager, **I want to** improve the inventory management system **so that** I can track stock levels more accurately and streamline the restocking process.\n"
    "- **Acceptance Criteria**:\n"
    "  1. The system should provide real-time updates on stock levels as items are added or removed from inventory.\n"
    "  2. The system should generate automatic restocking alerts when stock levels fall below predefined thresholds.\n"
    "  3. The system should integrate with the company's ERP system to ensure consistency across all business operations.\n"
    "  4. The system should allow warehouse managers to view inventory levels, restocking alerts, and historical data through a user-friendly interface.\n"
    "  5. The system should support barcode scanning for efficient inventory management.\n"
    "- **Success Metrics**:\n"
    "  - Reduction in stock discrepancies by 80%.\n"
    "  - Decrease in restocking delays by 70%.\n"
    "  - Positive feedback from warehouse managers on the accuracy and usability of the system.\n\n"
    "Here is the component (including all embedded components):\n\n"
    f"{header_line}{full_expanded}"
)

                summary = claude.generate_text(user_prompt, system_prompt=system_prompt)
                summary_file = os.path.join(output_dir, f"{mrn_id}_summary.md")
                with open(summary_file, "w", encoding="utf-8") as sf: 
                    sf.write(header_line)   
                    sf.write(summary.strip() + "\n")
                logger.info(f"Saved summary for {mrn_id} to {summary_file}")
                print(f"Saved summary for {mrn_id} to {summary_file}")
            except Exception as e:
                logger.error(f"Failed to summarize MRN {mrn_id}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="CAGEN Backward Engineering - Expand and analyze COBOL components using Claude via AWS Bedrock."
    )
    parser.add_argument("--aws-access-key", help="AWS access key ID")
    parser.add_argument("--aws-secret-key", help="AWS secret access key")
    parser.add_argument("--profile", help="AWS profile name to use from credentials file")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--model", default="us.anthropic.claude-3-7-sonnet-20250219-v1:0", help="Claude model ID to use in Bedrock")
    parser.add_argument("--component-dir", default="set5", help="Directory containing component files")
    parser.add_argument("--output", default="documentation", help="Output directory for documentation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    global COMPONENT_DIR
    COMPONENT_DIR = args.component_dir

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Starting CAGEN Backward Engineering process")   
    logger.info(f"Model: {args.model}, Region: {args.region}")

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

        claude = BedrockClient(
            model_id=args.model,
            region=args.region,
            aws_access_key=args.aws_access_key,
            aws_secret_key=args.aws_secret_key
        )
        logger.info("Connected to AWS Bedrock and Claude model successfully.")

        entry_irns = detect_entry_irns()
        logger.info(f"Auto-detected entry IRNs: {entry_irns}")
        print(f"\nAuto-detected entry IRNs: {entry_irns}")

        for irn_id in entry_irns:
            process_irn(f"{irn_id}.txt", claude=claude, output_dir=output_dir)
        log_call_tree()
        logger.info("Component call tree logged.")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
