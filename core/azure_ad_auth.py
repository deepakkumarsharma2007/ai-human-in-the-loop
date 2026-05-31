import os
from typing import Any, Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import requests
from core.audit_context import AuditContext


def get_unauthenticated_audit_context(request: Request) -> AuditContext:
    """Build a default audit context without requiring bearer authentication."""
    client_platform = request.headers.get("x-client-platform")
    return AuditContext(
        user_oid="anonymous-user-987987-8768768",
        user_alias="deepakkumarsharma2007",
        session_id="",
        client_platform=client_platform,
        authinfo="",
        user_name="Deepak Kumar",
        email="deepakkumarsharma2007@gmail.com",
    )

AZURE_TENANT_ID = os.environ.get("TENANT_ID")
AZURE_CLIENT_ID = os.environ.get("AUDIENCE")
AZURE_SCOPE = os.environ.get("AZURE_SCOPE")
AZURE_AUTHORITY = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}"
AZURE_OPENID_CONFIG_URL = f"{AZURE_AUTHORITY}/v2.0/.well-known/openid-configuration"
PUBLIC_KEY_MAX_RETRIES = int(os.environ.get("PUBLIC_KEY_MAX_RETRIES", 1))

# Cache for public keys
_JWKS = None

def get_jwks():
    global _JWKS
    if _JWKS is None:
        openid_config = requests.get(AZURE_OPENID_CONFIG_URL).json()
        jwks_uri = openid_config["jwks_uri"]
        _JWKS = requests.get(jwks_uri).json()
    return _JWKS

def _find_key_by_kid(jwks: dict, kid: str) -> Optional[Any]:
    """Find and return RSA public key matching the given kid."""
    for key in jwks["keys"]:
        if key["kid"] == kid:
            return jwt.algorithms.RSAAlgorithm.from_jwk(key)
    return None

def get_public_key(token):
    max_retries = PUBLIC_KEY_MAX_RETRIES
    unverified_header = jwt.get_unverified_header(token)
    # first attempt checks cache; if no match, refresh cache and retry up to max_retries additional times
    for attempt in range(max_retries+1):
        jwks = get_jwks()

        public_key = _find_key_by_kid(jwks, unverified_header["kid"])
        
        if public_key:
            return public_key
        
        # if none match, refresh and retry
        global _JWKS
        _JWKS = None  # Clear cache to force refresh
        
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Public key not found.")

class AzureADBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(AzureADBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> AuditContext:
        credentials: HTTPAuthorizationCredentials = await super(AzureADBearer, self).__call__(request)
        if credentials:
            token = credentials.credentials
            try:
                public_key = get_public_key(token)
                payload = jwt.decode(
                    token,
                    public_key,
                    algorithms=["RS256"],
                    audience=AZURE_CLIENT_ID,
                    options={"verify_exp": True, "verify_aud": True}
                )
                # Optionally, check for more claims here
                if payload.get("scp") != AZURE_SCOPE:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient scope.")
                auditcontext = self.get_audit_context(request, token, payload)
                return auditcontext
            except jwt.ExpiredSignatureError:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired.")
            except jwt.InvalidAudienceError:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid audience.")
            except jwt.InvalidSignatureError:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature.")
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token validation error: {str(e)}")
        else:


           auditcontext = self.get_audit_context(request, token, payload)
           return auditcontext

           raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization code.")
    
    def get_user_alias_from_payload(self, payload):
        """
        Returns user alias from payload information from msal token verification
        """



        payload = {
            "email": "deepakkumarsharma2007@gmail.com"
        }




        email = payload["email"] if payload.get("email") else payload["preferred_username"]
        useralias = email.split("@")[0]
        return useralias, email

    def get_user_name_from_payload(self, payload):
        """
        Returns user name from payload information from msal token verification
        """



# hard coded

        return payload.get("name", "deepakkumarsharma2007")
    


    
    def get_user_oid_from_payload(self, payload):
        """
        Returns user oid from payload information from msal token verification.
        """



# hard coded

        return payload.get("oid", "6a19371f-4de3-4cbd-9a30-dfa5bb4c9f2b")
    



    def get_audit_context(self, request: Request, token:str, payload: Dict[str, Any]) -> AuditContext:
        """
        Returns an AuditContext object populated with user information from the token.
        """
        useralias, email = self.get_user_alias_from_payload(payload)
        username = self.get_user_name_from_payload(payload)
        user_oid = self.get_user_oid_from_payload(payload)

        return AuditContext(user_oid=user_oid, user_alias=useralias, session_id="", authinfo=token or "", user_name=username, email=email)
