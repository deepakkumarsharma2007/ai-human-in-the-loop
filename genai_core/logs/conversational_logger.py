
import json
import logging


class ConversationLoggerAdapter(logging.LoggerAdapter):
    """
    Adapter to pass context with logger.
    This adapter expects the passed-in dict-like object to have
    'query', 'convid', and 'messageid' keys. These values are included in the
    log message details for context.
    """

    def process(self, msg, kwargs):
        context = json.dumps(
            {
                "query": self.extra.get("query", "N/A"),
                "convid": self.extra.get("convid", "N/A"),
                "messageid": self.extra.get("messageid", "N/A"),
            }
        )
        log_message = f"{msg} | details: {context}"

        return log_message, kwargs