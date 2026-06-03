import os

#### PROMPT_TO_CHECK_CAN_QUERY_BE_CACHED ###
PROMPT_TO_CHECK_CAN_QUERY_BE_CACHED = """You are an intelligent assistant. For each question-answer provided, we need to decide if the question would be relevant in the future. Think step by step. You need to give the output as dictionary with keys "decider" and "question". You need to understand the given question with the context of it's specific response.  Rephrase the question such that it clarifies the question based on it's specific context.The response may include the specific usecase of software/application. This should be the part of rephrased question.
Output format should be as follows:
{"decider": <YES/NO>, "question": <Rephrased question based on given question and corresponding response>}\n
Rules:
 
1. The question should not involve personal data or be user-specific.
2. The question should not be irrelevant or answerable with general knowledge.
3. The question should require technical knowledge to understand.
4. "decider" field should be YES or NO. This decides whether this question would be relevant in the future.
5. "question" field should be the rephrased question based on the given question and its response.
6. Rephrased question should be a valid question.
7. Rephrased question should be brief. Avoid unnecessary details.
 
Examples:
- Prompt: {"question": "What is the capital of India?","response":"The capital of India is New Delhi."}
  Output: {"decider": "NO", "question": "What is the capital of India?"}
- Prompt: {"question": "What are the privacy measures for XYZ platform?" "response": "The XYZ platform takes user privacy very seriously and has implemented several measures to ensure the protection of personal data. Here are some key privacy measures:\n\nData Encryption: All user data is encrypted both in transit and at rest using industry-"}
  Output: {"decider": "YES", "question": "What are the privacy measures for XYZ platform?"}
- Prompt: {"question":"What can you do?","response":"I can assist with a variety of tasks including:\n\n1. **Providing Information:**\n   - Answering questions about the XYZ software development platform using the XYZSearch tool.\n   - Retrieving usage statistics for the MyApp application using the GetMyAppUsageStatsTool.\n\n2. **Database Operations:**\n   - Executing find, insert, update, and delete operations on the XYZ application MongoDB database using the XYZSearch tool.\n\nIf you have any specific questions or tasks you'd like help with, feel free to ask!"}
  output: {"decider": "YES", "question": "What functionalities does XYZ provide?"}
 
Now, decide the answer for the following prompt:"""

if os.environ.get("REDIS_CACHE_SAVING_CHECK_PROMPTNAME") is not None:
  prompt_name = os.environ.get("REDIS_CACHE_SAVING_CHECK_PROMPTNAME")
  # agent_prompt = PromptManager().get_prompt(prompt_name)
  PROMPT_TO_CHECK_CAN_QUERY_BE_CACHED = "Cannot be cached"#agent_prompt.template.format({})

### REDIS_REPHRASAL_PROMPT_TEMPLATE ###

REDIS_REPHRASAL_PROMPT_TEMPLATE = "Summarize the following answer in few sentences or briefly for the user query {last_human_message_content}: {response_from_cache}"

if os.environ.get("REDIS_REPHRASAL_PROMPT_TEMPLATE") is not None:
  prompt_name = os.environ.get("REDIS_REPHRASAL_PROMPT_TEMPLATE")
  # agent_prompt = PromptManager().get_prompt(prompt_name)
  REDIS_REPHRASAL_PROMPT_TEMPLATE = "Cannot be cached"#agent_prompt.template