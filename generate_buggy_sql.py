#!/usr/bin/env python3
"""
SQL Buggy Query Generator

This script uses HuggingFace LLMs to generate incorrect SQL queries based on correct examples.
It takes a natural language request, database schema information, and a correct SQL query,
then generates multiple buggy variations with both logical and syntactic errors.

Usage:
    python generate_buggy_sql.py \
        --input-json data/sql_example.json \
        --output-json data/buggy_sql_output.json \
        --model-name microsoft/Phi-3-mini-4k-instruct \
        --num-buggy-queries 10 \
        --batch-size 8

Input JSON format (single example):
    {
        "user_request": "Find all users older than 18",
        "table_name": "users",
        "database": "mydb",
        "columns": ["id", "name", "age", "email"],
        "correct_query": "SELECT * FROM users WHERE age > 18"
    }

Output JSON format (array of buggy queries):
    [
        {
            "user_request": "Find all users older than 18",
            "table_name": "users",
            "database": "mydb",
            "columns": ["id", "name", "age", "email"],
            "correct_query": "SELECT * FROM users WHERE age > 18",
            "buggy_query": "SELECT * FROM user WHERE age >= 18",
            "error_type": "logical",
            "generation_params": {"temperature": 0.8, "model": "microsoft/Phi-3-mini-4k-instruct"}
        },
        ...
    ]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from tqdm import tqdm


@dataclass
class BuggySQLResult:
    """Result container for a single buggy SQL query generation."""
    user_request: str
    table_name: str
    database: str
    columns: List[str]
    correct_query: str
    buggy_query: str
    error_type: str
    generation_params: Dict[str, Any]


class SQLBuggyQueryGenerator:
    """
    Generator for creating buggy SQL queries using HuggingFace LLMs.

    Supports both instruction-following models (e.g., Phi, Mistral) and
    code generation models (e.g., DeepSeek-Coder, CodeLlama).
    """

    # Different prompt templates for variety in error types
    ERROR_PROMPTS = {
        "logical_comparison": "Generate an incorrect SQL query by using the wrong comparison operator (e.g., >= instead of >, != instead of =, etc.)",
        "logical_join": "Generate an incorrect SQL query by using the wrong type of JOIN or joining on incorrect columns",
        "logical_aggregation": "Generate an incorrect SQL query by using the wrong aggregation function or missing GROUP BY/HAVING clauses",
        "logical_column": "Generate an incorrect SQL query by selecting wrong columns or using incorrect column names",
        "logical_order": "Generate an incorrect SQL query by using incorrect ORDER BY or missing necessary sorting",
        "syntactic_table": "Generate an incorrect SQL query with a syntax error in the table name (e.g., missing quotes, typo)",
        "syntactic_keyword": "Generate an incorrect SQL query with a syntax error in SQL keywords (e.g., SELCT, FORM, WERE)",
        "syntactic_punctuation": "Generate an incorrect SQL query with syntax errors in punctuation (e.g., missing commas, extra semicolons, wrong parentheses)",
        "syntactic_structure": "Generate an incorrect SQL query with structural syntax errors (e.g., wrong clause order, missing FROM clause)",
        "mixed": "Generate an incorrect SQL query with both logical and syntactic errors",
    }

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: Optional[str] = None,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
    ):
        """
        Initialize the SQL Buggy Query Generator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cuda/cpu), auto-detected if None
            temperature: Temperature for generation (higher = more varied)
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",  # For generation
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate dtype
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        try:
            # Try loading as causal LM first (most common for instruction models)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model_type = "causal_lm"
            print("Loaded as CausalLM")

        except Exception as e:
            print(f"Failed to load as CausalLM: {e}")
            try:
                # Try seq2seq models (e.g., T5, BART)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=self.device if self.device == "cuda" else None,
                    trust_remote_code=True,
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                self.model_type = "seq2seq_lm"
                print("Loaded as Seq2SeqLM")

            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model {model_name} as either CausalLM or Seq2SeqLM. "
                    f"Errors: {e}, {e2}"
                )

        self.model.eval()
        print("Model loaded successfully!\n")

    def _format_prompt(
        self,
        user_request: str,
        table_name: str,
        database: str,
        columns: List[str],
        correct_query: str,
        error_instruction: str,
    ) -> str:
        """
        Format the prompt for the LLM to generate a buggy SQL query.

        Args:
            user_request: Natural language request
            table_name: Database table name
            database: Database name
            columns: List of column names
            correct_query: The correct SQL query
            error_instruction: Specific instruction for what type of error to introduce

        Returns:
            Formatted prompt string
        """
        columns_str = ", ".join(columns)

        prompt = f"""You are a SQL expert tasked with generating INCORRECT SQL queries for testing purposes.

Given the following information:
- User Request: {user_request}
- Database: {database}
- Table: {table_name}
- Columns: {columns_str}
- Correct SQL Query: {correct_query}

Task: {error_instruction}

IMPORTANT:
- Generate ONLY the buggy SQL query, nothing else
- Do NOT include explanations or comments
- The query should be a modification of the correct query
- Make the error subtle but meaningful
- Ensure the query is related to the original request but incorrect

Buggy SQL Query:"""

        return prompt

    def _extract_sql_from_generation(self, generated_text: str, prompt: str) -> str:
        """
        Extract the SQL query from the generated text.

        Args:
            generated_text: Full generated text from the model
            prompt: Original prompt (to remove it from output)

        Returns:
            Extracted SQL query
        """
        # Remove the prompt from the generated text
        if prompt in generated_text:
            sql = generated_text.split(prompt)[-1]
        else:
            sql = generated_text

        # Clean up the SQL query
        sql = sql.strip()

        # Remove common unwanted prefixes/suffixes
        unwanted_prefixes = ["```sql", "```", "SQL:", "Query:", "Answer:"]
        for prefix in unwanted_prefixes:
            if sql.startswith(prefix):
                sql = sql[len(prefix):].strip()

        unwanted_suffixes = ["```", ";", "END"]
        for suffix in unwanted_suffixes:
            if sql.endswith(suffix):
                sql = sql[:-len(suffix)].strip()

        # If there are multiple lines, take the first non-empty line
        # (often the SQL query is on the first line)
        lines = [line.strip() for line in sql.split('\n') if line.strip()]
        if lines:
            sql = lines[0]

        # Remove trailing semicolon for consistency (we can add it back later if needed)
        sql = sql.rstrip(';').strip()

        return sql

    def generate_single_buggy_query(
        self,
        user_request: str,
        table_name: str,
        database: str,
        columns: List[str],
        correct_query: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single buggy SQL query.

        Args:
            user_request: Natural language request
            table_name: Database table name
            database: Database name
            columns: List of column names
            correct_query: The correct SQL query
            error_type: Specific error type to generate (random if None)

        Returns:
            Dictionary with buggy query and metadata
        """
        # Select error type
        if error_type is None:
            import random
            error_type = random.choice(list(self.ERROR_PROMPTS.keys()))

        error_instruction = self.ERROR_PROMPTS[error_type]

        # Format prompt
        prompt = self._format_prompt(
            user_request=user_request,
            table_name=table_name,
            database=database,
            columns=columns,
            correct_query=correct_query,
            error_instruction=error_instruction,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract SQL query
        buggy_query = self._extract_sql_from_generation(generated_text, prompt)

        # Determine if error is logical or syntactic based on error_type
        error_category = "syntactic" if "syntactic" in error_type else "logical"
        if "mixed" in error_type:
            error_category = "mixed"

        return {
            "buggy_query": buggy_query,
            "error_type": error_category,
            "specific_error": error_type,
            "generation_params": {
                "temperature": self.temperature,
                "model": self.model_name,
                "max_new_tokens": self.max_new_tokens,
            }
        }

    def generate_buggy_queries(
        self,
        user_request: str,
        table_name: str,
        database: str,
        columns: List[str],
        correct_query: str,
        num_queries: int = 5,
        batch_size: int = 8,
    ) -> List[BuggySQLResult]:
        """
        Generate multiple buggy SQL queries using batch processing for efficiency.

        Args:
            user_request: Natural language request
            table_name: Database table name
            database: Database name
            columns: List of column names
            correct_query: The correct SQL query
            num_queries: Number of buggy queries to generate
            batch_size: Number of queries to generate in parallel

        Returns:
            List of BuggySQLResult objects
        """
        # Distribute error types to ensure variety
        error_types = list(self.ERROR_PROMPTS.keys())
        error_types_to_use = []

        # Cycle through error types to ensure variety
        for i in range(num_queries):
            error_types_to_use.append(error_types[i % len(error_types)])

        # Prepare all prompts in advance
        print(f"Preparing {num_queries} prompts...")
        prompts_and_metadata = []

        for error_type in error_types_to_use:
            error_instruction = self.ERROR_PROMPTS[error_type]
            prompt = self._format_prompt(
                user_request=user_request,
                table_name=table_name,
                database=database,
                columns=columns,
                correct_query=correct_query,
                error_instruction=error_instruction,
            )

            # Determine error category
            error_category = "syntactic" if "syntactic" in error_type else "logical"
            if "mixed" in error_type:
                error_category = "mixed"

            prompts_and_metadata.append({
                "prompt": prompt,
                "error_type": error_category,
                "specific_error": error_type,
            })

        # Create collate function for DataLoader
        def collate_fn(batch):
            """Collate function to tokenize batches of prompts."""
            prompts = [item["prompt"] for item in batch]
            tokenized = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            return tokenized, batch

        # Generate in batches
        results = []
        dataloader = DataLoader(
            prompts_and_metadata,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

        print(f"Generating {num_queries} buggy SQL queries in batches of {batch_size}...")

        with torch.no_grad():
            for batch_tokenized, batch_metadata in tqdm(dataloader, desc="Generating batches"):
                try:
                    # Move inputs to device
                    input_ids = batch_tokenized["input_ids"].to(self.device)
                    attention_mask = batch_tokenized["attention_mask"].to(self.device)

                    # Generate
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True if self.temperature > 0 else False,
                        top_p=0.95,
                        top_k=50,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decode all generated queries in the batch
                    decoded_texts = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )

                    # Process each generated query
                    for decoded_text, metadata in zip(decoded_texts, batch_metadata):
                        # Extract SQL query
                        buggy_query = self._extract_sql_from_generation(
                            decoded_text,
                            metadata["prompt"]
                        )

                        # Create result object
                        result = BuggySQLResult(
                            user_request=user_request,
                            table_name=table_name,
                            database=database,
                            columns=columns,
                            correct_query=correct_query,
                            buggy_query=buggy_query,
                            error_type=metadata["error_type"],
                            generation_params={
                                "temperature": self.temperature,
                                "model": self.model_name,
                                "max_new_tokens": self.max_new_tokens,
                                "batch_size": batch_size,
                            }
                        )

                        results.append(result)

                except Exception as e:
                    print(f"\nWarning: Failed to generate batch: {e}")
                    continue

        print(f"\nSuccessfully generated {len(results)}/{num_queries} buggy queries")
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate buggy SQL queries using HuggingFace LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default model
  python generate_buggy_sql.py --input-json data/sql_example.json --num-buggy-queries 10

  # With specific model, batch size and output
  python generate_buggy_sql.py \\
    --input-json data/sql_example.json \\
    --output-json data/buggy_sql_output.json \\
    --model-name deepseek-ai/deepseek-coder-1.3b-instruct \\
    --num-buggy-queries 20 \\
    --batch-size 4 \\
    --temperature 0.9

  # Using CPU only with smaller batch
  python generate_buggy_sql.py \\
    --input-json data/sql_example.json \\
    --device cpu \\
    --num-buggy-queries 5 \\
    --batch-size 2
        """
    )

    # Required arguments
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to input JSON file containing SQL example"
    )

    # Optional arguments
    parser.add_argument(
        "--output-json",
        type=str,
        default="buggy_sql_output.json",
        help="Path to output JSON file (default: buggy_sql_output.json)"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="HuggingFace model name (default: microsoft/Phi-3-mini-4k-instruct)"
    )

    parser.add_argument(
        "--num-buggy-queries",
        type=int,
        default=5,
        help="Number of buggy queries to generate (default: 5)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for generation (default: 0.8)"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for parallel generation (default: 8)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load input JSON
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading input from: {input_path}")
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    # Validate input data
    required_fields = ["user_request", "table_name", "database", "columns", "correct_query"]
    for field in required_fields:
        if field not in input_data:
            print(f"Error: Missing required field in input JSON: {field}")
            sys.exit(1)

    print(f"\nInput data:")
    print(f"  User Request: {input_data['user_request']}")
    print(f"  Database: {input_data['database']}")
    print(f"  Table: {input_data['table_name']}")
    print(f"  Columns: {', '.join(input_data['columns'])}")
    print(f"  Correct Query: {input_data['correct_query']}")
    print()

    # Initialize generator
    generator = SQLBuggyQueryGenerator(
        model_name=args.model_name,
        device=args.device,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Generate buggy queries
    results = generator.generate_buggy_queries(
        user_request=input_data["user_request"],
        table_name=input_data["table_name"],
        database=input_data["database"],
        columns=input_data["columns"],
        correct_query=input_data["correct_query"],
        num_queries=args.num_buggy_queries,
        batch_size=args.batch_size,
    )

    # Convert to list of dicts for JSON serialization
    output_data = [asdict(result) for result in results]

    # Save output JSON
    output_path = Path(args.output_json)
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDone! Generated {len(results)} buggy SQL queries.")
    print(f"Output saved to: {output_path}")

    # Print summary
    print("\nError type distribution:")
    error_counts = {}
    for result in results:
        error_type = result.error_type
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

    for error_type, count in sorted(error_counts.items()):
        print(f"  {error_type}: {count}")

    # Print a few examples
    print("\nExample buggy queries:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n  {i}. [{result.error_type}] {result.buggy_query}")


if __name__ == "__main__":
    main()
