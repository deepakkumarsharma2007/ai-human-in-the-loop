import json
from typing import Any, Dict, List
from langchain.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import ToolMessage
from models.parts import DataPart


def get_artifacts_from_messages(messages: List[BaseMessage]):
    """
    Extracts artifacts from messages and combines them together.
    Returns:
        tuple: A tuple containing:
            - artifacts (list): Regular artifacts from messages.
            - response_artifacts (list): Artifacts marked as agent responses.
    """
    artifacts = []
    response_artifacts = []
    for message in messages:

        if not isinstance(message, ToolMessage):
            continue

        if not getattr(message, "artifact", None):
            continue

        if not isinstance(message.artifact, list):
            continue

        for artifact_item in message.artifact:
            if hasattr(artifact_item, "metadata"):
                if "agent_response" in artifact_item.metadata.get("source", ""):
                    response_artifacts.append(artifact_item)
                    continue

            artifacts.append(artifact_item)

    return artifacts, response_artifacts


def get_references_response_artifact_parts(chat_response_references: List[Dict]):

    # Get artifacts and response artifacts from chat response references
    chat_response_references_ = [
        ref
        for ref in chat_response_references
        if ref.get("type") != "response_artifacts"
    ]
    response_artifacts = [
        ref
        for ref in chat_response_references
        if ref.get("type") == "response_artifacts"
    ]

    # extract part from response artifacts
    response_artifact_parts = []
    for response_artifact in response_artifacts:
        resp_artifacts = response_artifact["artifacts"]
        for resp_artifact in resp_artifacts:
            response_artifact_parts = response_artifact_parts + getattr(
                resp_artifact, "parts", []
            )

    return chat_response_references_, response_artifact_parts


def get_execution_path_items_from_messages(messages: List[BaseMessage]):
    """ "
    Get execution path from state
    """
    execution_path_items = []

    # FOr every message in messages
    # Process for extracting execution path items
    for message in messages:

        # Only process message with tool calls
        if not getattr(message, "tool_calls", None):
            continue

        # For each message with tool call extract tool call
        # As tool calls are execution path
        for tool_call in message.tool_calls:
            tool_args = tool_call["args"].copy()

            # Remove the auditcontext and shared tool inputs form tool args as
            # Not required
            if tool_args.get("auditcontext"):
                del tool_args["auditcontext"]
            if tool_args.get("sharedtoolinputs"):
                del tool_args["sharedtoolinputs"]

            # Add if agent is registerd
            if tool_call["name"]:
                execution_path_items.append(
                    {"name": tool_call["name"], "info": {"args": tool_args}}
                )

    # Return the prepared execution path items
    return execution_path_items


def get_tags_from_context_and_response(auditContext, response_cits):
    """
    Private method to get tags from auditContext and response_cits.
    Returns a list of tags.
    """
    tags = []
    response_tags = get_tags_from_execution_path(response_cits)
    if response_tags:
        tags.extend(response_tags)
    if (
        auditContext
        and isinstance(auditContext, dict)
        and auditContext.get("client_platform")
    ):
        tags.append(f"Platform:{auditContext['client_platform']}")
    return tags


def get_reference_weblinks_from_message(message: BaseMessage) -> list[dict]:
    """
    Parses message content as json.

    Args:
        message_content (str): Message content to parse and get references from.

    Returns:
        list[dict]: List of dictionaries where each dictionary representing reference.
    """
    # Parse JSON, fall back to empty references on error.

    # Try decoding as json
    try:
        content_json = json.loads(message.content)
    except json.JSONDecodeError as e:
        return []

    # Handle error in processing references
    try:
        if type(content_json) != list:
            return []

        content_json_msgs = content_json
        references = []

        for content_json_msg in content_json_msgs:
            if (
                content_json_msg.get("result_type") != "json"
                or not content_json_msg.get("references")
                or not type(content_json_msg.get("references")) == list
            ):
                continue
            else:
                references = references + content_json_msg.get("references", [])

        # Remove duplicates
        references = list({r["title"]: r for r in references}.values())

    except Exception as e:
        return []

    return references


def get_reference_weblinks_from_messages(messages: List[BaseMessage]):
    """
    Get reference weblinks from messages.

    """
    weblinks = []
    for message in messages:
        if type(message) != ToolMessage:
            continue
        weblinks.append(get_reference_weblinks_from_message(message))

    # Need to merge weblinks as it is list of lists
    if len(weblinks) == 0:
        return weblinks

    weblinks_merged = [item for sublist in weblinks for item in sublist]

    # Remove duplicates from merged list
    weblinks_unique = list(
        {weblink["title"]: weblink for weblink in weblinks_merged}.values()
    )

    return weblinks_unique


def get_references_from_messages(
    messages: list[dict], message_limit: int = 100
) -> list[dict]:
    """
    Returns references from messages. Processes messages in reverse order, last first.

    Args:
        messages (list[dict]): List of dictionaries where each dictionary represents message - ToolMessage/AIMessage/HumanMessage.
        message_limit (int): No. of entries to process for retriveing references.

    Return:
        list[dict]: List of dictionaries where each dictionary represents reference.

    """

    # Set message to process
    # As messages are revered and we need to get in limit window
    messages_to_process = []
    for message_index, message in enumerate(reversed(messages)):
        if message_index > message_limit:
            break
        if isinstance(message, HumanMessage):
            break
        messages_to_process.append(message)

    # Reverse to reset original order
    messages_to_process = list(reversed(messages))

    # Get web links and execution path items from message to process
    web_links = get_reference_weblinks_from_messages(messages_to_process)
    execution_path_items = get_execution_path_items_from_messages(messages_to_process)
    artifacts_from_messages, response_artifacts = get_artifacts_from_messages(
        messages_to_process
    )

    return [
        {"type": "web_links", "links": web_links},
        {"type": "execution_path", "agents": execution_path_items},
        {"type": "artifacts", "artifacts": artifacts_from_messages},
        {"type": "response_artifacts", "artifacts": response_artifacts},
    ]


def get_document_references_from_datapart(datapart: DataPart) -> dict | None:
    """
    Extracts document references from a DataPart object.

    Args:
        datapart (DataPart): The DataPart object to extract references from.

    Returns:
        DocumentReferenceModel: A model containing document references.
    """
    document_references = {"documentIds": [], "documentFiles": []}
    if datapart and datapart.data:
        document_ids = datapart.data.get("documentIds", [])
        document_files = datapart.data.get("documentFiles", [])
        if document_ids or document_files:
            document_references = {
                "documentIds": document_ids,
                "documentFiles": document_files,
            }
            return {"type": "document", "data": document_references}
    return None


def get_document_references_from_data(dataval: Dict[str, Any]) -> dict | None:
    """
    Extracts document references from a data dictionary.

    Args:
        dataval (Dict[str, Any]): The data dictionary to extract references from.

    Returns:
        dict | None: A dictionary containing document references, or None if no references are found.
    """
    document_references = {"documentIds": [], "documentFiles": []}
    if dataval:
        document_ids = dataval.get("documentIds", [])
        document_files = dataval.get("documentFiles", [])
        if document_ids or document_files:
            document_references = {
                "documentIds": document_ids,
                "documentFiles": document_files,
            }
            return {"type": "document", "data": document_references}
    return None


def get_tags_from_execution_path(references: Any) -> list[str] | None:
    """
    Returns a list of tags from the execution path items.
    Each tag is formatted as "Agent:{name}" where name is from the execution path item.

    Args:
        references (Any): List of reference dictionaries, expected to include an 'execution_path' type.

    Returns:
        list[str] | None: List of tags, or None if not found.
    """
    if references:
        execution_path_cits = [
            cit for cit in references if cit.get("type") == "execution_path"
        ]
        if execution_path_cits and isinstance(execution_path_cits[0], dict):
            executed_agents = execution_path_cits[0].get("agents", [])
            tags = []
            for agent in executed_agents:
                name = agent.get("name")
                if name:
                    tags.append(f"Agent:{name}")
            return tags
        return None
    return None


import re
import json
from typing import Any, Tuple

def can_parse_to_summary_response(data: Any) -> Any:
    try:
        if not isinstance(data, dict):
            return data
        
        if all(key in data for key in ["answer", "hasemailreference", "reasoning"]):
            return data  # Already valid

        # If only 'answer' exists and others are embedded inside it
        full_answer = data.get("answer", "")
        email_ref_match = re.search(r'hasemailreference:\s*["\']?(.*?)["\']?(?=\s|$|\n)', full_answer, flags=re.DOTALL | re.IGNORECASE)
        agentsused_match = re.search(r'agentsused:\s*(\[.*?\])', full_answer, flags=re.DOTALL | re.IGNORECASE)
        reasoning_match = re.search(r'reasoning:\s*["\']?(.*?)["\']?(?=\s|$|\n)', full_answer, flags=re.DOTALL | re.IGNORECASE)

        if email_ref_match and reasoning_match:
            hasemailreference = email_ref_match.group(1).strip('\\"')
            # Parse agentsused as JSON array
            try:
                agentsused = json.loads(agentsused_match.group(1)) if agentsused_match else []
            except:
                agentsused = []
            reasoning = reasoning_match.group(1).strip('\\"')

            # Remove embedded fields from answer
            # Find the first occurrence of any field marker and take everything before it
            pattern = r'(?:hasemailreference|agentsused|reasoning)\s*:'
            match = re.search(pattern, full_answer, flags=re.IGNORECASE)
            if match:
                cleaned_answer = full_answer[:match.start()].strip()
            else:
                cleaned_answer = full_answer.strip()

            return {
                "answer": cleaned_answer,
                "hasemailreference": hasemailreference,
                "agentsused": agentsused,
                "reasoning": reasoning
            }
        return data
    except Exception:
        return data

def extract_json_or_answer(raw: str) -> Tuple[str, bool]:
    """
    Attempts to parse input into JSON.
    
    Returns:
    - result string (JSON string if parsed, else answer via regex, else raw string)
    - is_json flag: True if JSON parsing succeeded, False otherwise
    """
    if not isinstance(raw, str):
        raise ValueError("Input must be a string")

    # Remove backticks / ```json fences
    clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    # Step 1: Try normal JSON parsing
    try:
        parsed = json.loads(clean)
        validresponse = can_parse_to_summary_response(parsed)
        return json.dumps(validresponse, ensure_ascii=False), True
    except json.JSONDecodeError:
        try:
            # Try double-decoding (JSON string containing JSON text)
            parsed = json.loads(json.loads(clean))
            return json.dumps(parsed, ensure_ascii=False), True
        except Exception:
            pass

    # Step 2: Regex fallback → get "answer"
    pattern = r'["\']answer["\']\s*:\s*["\']((?:\\.|[^"\'])*?)["\']'
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        value = match.group(1)
        try:
            value = value.encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            value = value.strip()
        return value.strip(), False

    # Step 3: Final fallback → return the raw string
    return raw.strip(), False