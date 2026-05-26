import uuid

class AuditContext:
    def __init__(self, user_oid:str, user_alias: str, session_id: str, client_platform: str | None = None, **kwargs):
        self.user_oid = user_oid
        self.user_alias = user_alias
        self.session_id = session_id
        self.transaction_id = str(uuid.uuid4())
        self.references: list[dict] = []    
        self.client_platform: str | None = client_platform
        self.additional_args = kwargs

    def __repr__(self):
        return (f"AuditContext(user_alias={self.user_alias}, "
                f"user_oid={self.user_oid},"
                f"session_id={self.session_id}, additional_args={self.additional_args}, "
                f"references={self.references}, client_platform={self.client_platform})")
        
    def to_dict(self):
        return {
            "user_oid": self.user_oid,
            "user_alias": self.user_alias,
            "session_id": self.session_id,
            "transaction_id": self.transaction_id,
            "additional_args": dict(self.additional_args),
            "references": self.references,
            "client_platform": self.client_platform
        }