"""GenAILens module for LangSmith integration with PII redaction."""

import ast
import importlib.util
import json
import os
import re
from functools import lru_cache
from typing import Optional

from langchain_classic.smith import RunEvalConfig, run_on_dataset
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai


def is_truthy_env(value: Optional[str]) -> bool:
    """Parse common truthy environment values."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

# Regex patterns for detecting and redacting sensitive information
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
UUID_REGEX = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-" r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
USER_NAME_REGEX = re.compile(r"user\s*name\s*:[^\n\r]+", re.IGNORECASE)

# Sensitive keys that should always be redacted
SENSITIVE_KEYS = frozenset(
    {
        "user_alias",
        "useralias",
        "audit_context",
        "auditContext",
        "auditcontext",
        "username",
        "User Name",
        "UserName",
        "Username",
    }
)

# Keys that should preserve content structure for tokenization
EXEMPT_STRING_KEYS = frozenset({"prompt", "system", "assistant", "name", "role"})


def apply_regex_redaction(text: str) -> str:
    """Apply regex-based redaction to preserve text structure."""
    text = re.sub(EMAIL_REGEX, "<email-address>", text)
    text = re.sub(UUID_REGEX, "<UUID>", text)
    text = re.sub(USER_NAME_REGEX, "User Name: <user-name>", text)
    return text


def update_tool_call_args(data_dict: dict, tools_list: list) -> bool:
    """
    Redact sensitive tool call arguments in LLMResult structures.
    Preserves overall data structure while masking sensitive content.
    Returns True if any redaction occurred.
    """
    if data_dict.get("type") != "LLMResult":
        return False

    generations = data_dict.get("generations")
    if not isinstance(generations, list):
        return False

    changed = False
    try:
        # Iterate through nested generation structure
        for gen_group in generations:
            if not isinstance(gen_group, list):
                continue
            changed |= redact_generation_group(gen_group, tools_list)
    except (KeyError, TypeError, ValueError):
        # Malformed or unexpected generation/tool_call structure; ignore and report no changes.
        pass

    return changed


def redact_generation_group(gen_group: list, tools_list: list) -> bool:
    """Helper function to redact tool calls in a generation group."""
    changed = False
    for gen in gen_group:
        message_kwargs = gen.get("message", {}).get("kwargs", {})
        tool_calls = message_kwargs.get("tool_calls")

        if not isinstance(tool_calls, list):
            continue

        changed |= redact_tool_calls(tool_calls, tools_list)
    return changed


def redact_tool_calls(tool_calls: list, tools_list: list) -> bool:
    """Helper function to redact arguments in tool calls."""
    changed = False
    for tool in tool_calls:
        name = tool.get("name") or tool.get("function", {}).get("name")
        if name and name in tools_list and "args" in tool:
            changed = True
    return changed


def replace_sensitive_data(
    data,
    depth: int = 20,
    tools: list = None,
    force_redact: bool = False,
):
    """
    Recursively redact sensitive data in nested structures.
    Preserves data types and structure for LLM tokenization compatibility.

    Args:
        data: Input data structure to redact
        depth: Recursion depth limit
        tools: List of sensitive tool names
        force_redact: Force redaction of all strings
    """
    if depth == 0:
        return data

    if tools is None:
        tools = [
            "CreatePowerpointAdapter",
            "DraftMeeting",
            "ReadEmails",
            "DraftEmail",
        ]

    # Detect redaction flags in data
    data_str = str(data)
    # Consolidated check for redact/hasemailreference flags, including escaped forms
    flag_redact = bool(
        re.search(
            r"""(\\+)?['"]?redact(\\+)?['"]?\s*(\\*)?[:\\](\\*)?\s*(\\*)?(true|['"]true['"])""",
            data_str,
            re.IGNORECASE,
        )
    )
    if flag_redact and not re.search(r"lc:1", data_str):
        return {"output": "[redacted]"}
    flag_email_ref = bool(
        re.search(
            r"""['"]?hasemailreference['"]?\s*\\*:\s*\\*(?:true|['"]true['"])""",
            data_str,
            re.IGNORECASE,
        )
    )
    force_redact = force_redact or flag_redact or flag_email_ref

    # Check if agentsused field contains any of the tools in the list
    agents_used_match = re.search(
        r"""['"]?agentsused['"]?\s*:\s*\[(.*?)\]""", data_str, re.IGNORECASE
    )
    if agents_used_match:
        agents_content = agents_used_match.group(1)
        # Extract agent names from the array
        agent_names = re.findall(r"""['"]([^'"]+)['"]""", agents_content)
        # Check if any agent in the list matches tools
        if any(agent in tools for agent in agent_names):
            return {"output": "[redacted]"}

    # Handle tool call redaction for dict structures
    if isinstance(data, dict):
        update_tool_call_args(data, tools)

        redacted_dict = {}
        for k, v in data.items():
            # Always redact sensitive keys
            if k.lower() in SENSITIVE_KEYS:
                redacted_dict[k] = (
                    "[redacted]"
                    if isinstance(v, str)
                    else replace_sensitive_data(v, depth - 1, tools, force_redact=True)
                )
                continue

            # Apply forced redaction selectively
            if force_redact and isinstance(v, str):
                # Preserve structure for tokenizable fields
                redacted_dict[k] = (
                    apply_regex_redaction(v)
                    if k in EXEMPT_STRING_KEYS
                    else "[redacted]"
                )
            else:
                redacted_dict[k] = replace_sensitive_data(
                    v, depth - 1, tools, force_redact
                )
        return redacted_dict

    if isinstance(data, list):
        # Redact list items recursively
        redacted_list = []
        for item in data:
            if force_redact and isinstance(item, str):
                redacted_list.append(apply_regex_redaction(item))
            else:
                redacted_list.append(
                    replace_sensitive_data(item, depth - 1, tools, force_redact)
                )
        return redacted_list

    if isinstance(data, str):
        # Attempt to parse and redact structured strings
        if any(key in data for key in SENSITIVE_KEYS):
            try:
                parsed = ast.literal_eval(data)
                if isinstance(parsed, dict):
                    parsed = replace_sensitive_data(
                        parsed, depth - 1, tools, force_redact
                    )
                    return json.dumps(parsed)
            except (ValueError, SyntaxError):
                # If parsing fails, fall through to the standard string redaction logic below.
                pass

        # Apply string-level redaction
        return "[redacted]" if force_redact else apply_regex_redaction(data)

    return data


# Load sensitive tools from environment at module initialization
SENSITIVE_TOOLS_ENV = os.environ.get("SENSITIVE_TOOLS", "")
SENSITIVE_TOOLS = [
    tool.strip() for tool in SENSITIVE_TOOLS_ENV.split(",") if tool.strip()
]


@lru_cache(maxsize=1)
def get_genailens_client() -> Client:
    """
    Returns singleton GenAILens client with input/output redaction.
    Uses SENSITIVE_TOOLS environment variable for tool-specific filtering.
    """

    def hide_inputs(inputs):
        return replace_sensitive_data(inputs, tools=SENSITIVE_TOOLS)

    def hide_outputs(outputs):
        return replace_sensitive_data(outputs, tools=SENSITIVE_TOOLS)

    return Client(hide_inputs=hide_inputs, hide_outputs=hide_outputs)


class GenAILens:
    """
    LangSmith integration wrapper for tracing, monitoring,
    and evaluating LLM applications with automatic PII redaction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        enable_tracing: bool = False,
    ):
        """Initialize GenAILens with LangSmith configuration."""
        # self.api_key = api_key or os.environ.get("LANGCHAIN_API_KEY")
        # self.project_name = project_name or os.environ.get("LANGCHAIN_PROJECT")
        # self.enable_tracing = enable_tracing or os.environ.get("LANGCHAIN_TRACING")

        # Configure LangSmith environment
        # if self.api_key:
        #     os.environ["LANGCHAIN_API_KEY"] = self.api_key
        # if self.project_name:
        #     os.environ["LANGCHAIN_PROJECT"] = self.project_name
        # if self.enable_tracing:
        #     os.environ["LANGCHAIN_TRACING"] = "true"

        # os.environ["LANGSMITH_ENDPOINT"] = os.environ["AILENS_ENDPOINT"]

        # Verify dependencies and initialize client
        self.check_dependencies()
        self.client = get_genailens_client() if self._has_langsmith else None


    def check_dependencies(self):
        """Check availability of required packages."""
        self._has_langsmith = importlib.util.find_spec("langsmith") is not None
        self._has_openai = importlib.util.find_spec("openai") is not None
        self._has_langchain = importlib.util.find_spec("langchain") is not None

        if not self._has_langsmith:
            print(
                "Warning: langsmith package not found. "
                "Install with 'pip install langsmith'"
            )

    def wrap_openai(self, client):
        """Wrap OpenAI client to enable LangSmith tracing."""
        if not self._has_langsmith or not self._has_openai:
            print(
                "Warning: langsmith or openai package not found. "
                "Returning original client."
            )
            return client

        return wrap_openai(client)

    def traceable(self, func=None, name=None, run_type=None, tags=None, metadata=None):
        """Decorator for function-level tracing with LangSmith."""
        if not self._has_langsmith:

            def identity_decorator(f):
                return f

            return identity_decorator if func is None else identity_decorator(func)

        return traceable(
            client=self.client,
            func=func,
            name=name,
            run_type=run_type,
            tags=tags or [],
            metadata=metadata or {},
        )

    def create_project(self, project_name: str, description: str = ""):
        """Create new LangSmith project."""
        if not self._has_langsmith or not self.client:
            raise ImportError("langsmith package not found or client not initialized")
        return self.client.create_project(
            project_name=project_name, description=description
        )

    def create_dataset(self, dataset_name: str, description: str = ""):
        """Create new LangSmith dataset for evaluation."""
        if not self._has_langsmith or not self.client:
            raise ImportError("langsmith package not found or client not initialized")
        return self.client.create_dataset(
            dataset_name=dataset_name, description=description
        )

    def run_evaluation(
        self, dataset_name: str, llm_or_chain, evaluators=None, project_name=None
    ):
        """Execute evaluation run on specified dataset."""
        if not self._has_langsmith or not self._has_langchain:
            raise ImportError("langsmith or langchain package not found")

        eval_config = RunEvalConfig(evaluators=evaluators) if evaluators else None

        return run_on_dataset(
            client=self.client,
            dataset_name=dataset_name,
            llm_or_chain_factory=llm_or_chain,
            evaluation=eval_config,
            project_name=project_name or self.project_name,
        )