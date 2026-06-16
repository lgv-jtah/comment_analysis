import json
import sys
from string import Template
from collections import Counter, defaultdict
from pathlib import Path
"""Runs a stance detector"""
class StanceDetector:
    "Stance Detection with an LLM"
    def __init__(self):
        pass
    def detect_stance(self, text: str, target: str):
        return None

    def load_data(self):
        with open("gold_labels_corrected.json") as f:
            gold_labels = json.load(f)

        stance_counts = Counter(
            item.get("gold_stance")
            for item in gold_labels
            if item.get("gold_stance") is not None
        )
        # .get(...) is None catches both an explicit JSON null and an entirely missing key
        missing = sum(1 for item in gold_labels if item.get("gold_stance") is None)

        print("gold_stance distribution:")
        for label, count in stance_counts.most_common():
            print(f"  {label}: {count}")
        print(f"  missing/None: {missing}")

        return gold_labels


    def load_prompt_config(self, config_path: str, prompt_key: str) -> dict:
        """Load prompt config and resolve text file references.

        Config JSON maps prompt keys to file paths:
            {"base": {"system_prompt": "prompts/system.txt", "user_prompt": "prompts/user_base.txt"}}
        """
        config_dir = Path(config_path).parent

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if prompt_key not in config:
            print(
                f"Error: prompt key '{prompt_key}' not found. Available: {list(config.keys())}"
            )
            sys.exit(1)

        entry = config[prompt_key]

        # New format: file references
        if "system_prompt" in entry and "user_prompt" in entry:
            system_path = config_dir / entry["system_prompt"]
            user_path = config_dir / entry["user_prompt"]

            with open(system_path, "r", encoding="utf-8") as f:
                system_text = f.read().strip()
            with open(user_path, "r", encoding="utf-8") as f:
                user_template_text = f.read()

            return {
                "system": system_text,
                "user_template": user_template_text,
                "description": entry.get("description", ""),
                "format": "template",  # string.Template with $variables
            }

    def build_messages(self, prompt_config: dict, item: dict) -> tuple[str, list[dict]]:
        """Build system prompt and messages for a single stance evaluation.

        Supports both string.Template ($variable) and str.format_map ({variable}).
        """
        system = prompt_config["system"]

        # Map gold_labels.json field names to production template variable names
        template_vars = {
            # Production variable names
            "text_source": item.get("comment_text", ""),
            "key_message": item.get("key_message", ""),
            # Extended variable names (for non-production prompts)
            "comment_text": item.get("comment_text", ""),
            "project_id": item.get("project_id", ""),
            "category": item.get("category", ""),
            "contribution_text": item.get("contribution_text", ""),
            "key_message_index": str(item.get("key_message_index", "")),
            "key_message_type": item.get("key_message_type", ""),
        }

        if prompt_config.get("format") == "template":
            # string.Template with $variable — matches production code
            tmpl = Template(prompt_config["user_template"])
            user_content = tmpl.safe_substitute(template_vars)
        else:
            # str.format_map with {variable} — backwards compatible
            user_content = prompt_config["user_template"].format_map(
                defaultdict(str, template_vars)
            )

        messages = [{"role": "user", "content": user_content}]
        return system, messages
