import requests
import logging
from core.guardrails.guard_rail_service import GuardRailService
from core.guardrails.guard_rail_models import GuardRailsResponse, ScannerResult, LLMGuardRequestException

class LLMGuardRailService(GuardRailService):
    def __init__(self, logger = None):
        self.logger = logging.getLogger(__name__) if logger is None else logger

    def analyzeprompt(self, input_text: str) -> GuardRailsResponse:
        self.logger.info(f"Analyzing prompt... {input_text}")
        try:



            response = {} # build the service




            response_json = response

        except requests.RequestException as err:
            raise LLMGuardRequestException(err)

        isvalidscan:bool = True
        for key, value in response_json["valid"].items():
            if not value:  # If any value is False
                isvalidscan = False
                break
        scannerresults = [
            ScannerResult(type=key, value=str(value), message="")
            for key, value in response_json["score"].items()
            ]
        for scanner_result in scannerresults:
            self.logger.info(f"Scanner: {scanner_result.type}, Value: {scanner_result.value}, Message: {scanner_result.message}")
        
        return GuardRailsResponse(
            isvalid=isvalidscan,
            sanitizedstring=response_json["sanitizedinput"],
            scannerresults=scannerresults
        )

    def analyzeoutput(self, prompt: str, output: str) -> GuardRailsResponse:
        self.logger.info("Analyzing output...")
        try:
            response = {} # build the service

            response_json = response




        except requests.RequestException as err:
            raise LLMGuardRequestException(err)

        isvalidscan:bool = True
        for key, value in response_json["valid"].items():
            if not value:  # If any value is False
                isvalidscan = False
                break
        scannerresults = [
            ScannerResult(type=key, value=str(value), message="")
            for key, value in response_json["score"].items()
            ]
        for scanner_result in scannerresults:
            self.logger.info(f"Scanner: {scanner_result.type}, Value: {scanner_result.value}, Message: {scanner_result.message}")
        return GuardRailsResponse(
                isvalid=isvalidscan,
                sanitizedstring=response_json["sanitizedinput"],
                scannerresults=scannerresults
            )

    def scanprompt(self, input_text: str) -> GuardRailsResponse:
        self.logger.info("Scanning prompt...")
        try:
            response = {} # build the service

            response_json = response


        except requests.RequestException as err:
            raise LLMGuardRequestException(err)

        isvalidscan:bool = True
        for key, value in response_json["valid"].items():
            if not value:  # If any value is False
                isvalidscan = False
                break
        scannerresults = [
            ScannerResult(type=key, value=str(value), message="")
            for key, value in response_json["score"].items()
            ]
        for scanner_result in scannerresults:
            self.logger.info(f"Scanner: {scanner_result.type}, Value: {scanner_result.value}, Message: {scanner_result.message}")
                    
        return GuardRailsResponse(
            isvalid=isvalidscan,
            sanitizedstring=response_json["sanitizedinput"],
            scannerresults=scannerresults
        )

    def scanoutput(self, prompt: str, output: str) -> GuardRailsResponse:
        self.logger.info("Scanning output...")
        try:
            response = {} # build the service

            response_json = response


        except requests.RequestException as err:
            raise LLMGuardRequestException(err)

        isvalidscan:bool = True
        for key, value in response_json["valid"].items():
            if not value:  # If any value is False
                isvalidscan = False
                break
        scannerresults = [
            ScannerResult(type=key, value=str(value), message="")
            for key, value in response_json["score"].items()
            ]
        for scanner_result in scannerresults:
            self.logger.info(f"Scanner: {scanner_result.type}, Value: {scanner_result.value}, Message: {scanner_result.message}")            
        return GuardRailsResponse(
            isvalid=isvalidscan,
            sanitizedstring=response_json["sanitizedinput"],
            scannerresults=scannerresults
        )