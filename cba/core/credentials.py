import json
import os
from pathlib import Path
from typing import Dict, Optional

import keyring
from cryptography.fernet import Fernet

from .exceptions import CredentialError


class CredentialManager:
    def __init__(self, service_name: str = "cloud-billing-automation"):
        self.service_name = service_name
        self._encryption_key = None

    def _get_encryption_key(self) -> bytes:
        if self._encryption_key is not None:
            return self._encryption_key

        key_path = Path.home() / ".cba" / "encryption.key"
        key_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if key_path.exists():
                key = key_path.read_bytes()
                Fernet(key)
                self._encryption_key = key
            else:
                key = Fernet.generate_key()
                key_path.write_bytes(key)
                if os.name != "nt":
                    os.chmod(key_path, 0o600)
                self._encryption_key = key

            return self._encryption_key
        except Exception as exc:
            raise CredentialError(f"Failed to initialize credential encryption: {exc}") from exc

    def _encrypt_credential(self, credential: str) -> str:
        if not isinstance(credential, str):
            raise CredentialError("Credential value must be a string.")

        try:
            cipher = Fernet(self._get_encryption_key())
            return cipher.encrypt(credential.encode("utf-8")).decode("utf-8")
        except Exception as exc:
            raise CredentialError(f"Failed to encrypt credential: {exc}") from exc

    def _decrypt_credential(self, encrypted_credential: str) -> str:
        try:
            cipher = Fernet(self._get_encryption_key())
            return cipher.decrypt(
                encrypted_credential.encode("utf-8")
            ).decode("utf-8")
        except Exception as exc:
            raise CredentialError(f"Failed to decrypt credential: {exc}") from exc

    def _key(self, provider: str, credential_type: str) -> str:
        provider = str(provider).strip().lower()
        credential_type = str(credential_type).strip().lower()

        if not provider or not credential_type:
            raise CredentialError("Provider and credential type are required.")

        return f"{provider}_{credential_type}"

    def store_credential(
        self,
        provider: str,
        credential_type: str,
        value: str,
    ) -> None:
        if value is None or value == "":
            raise CredentialError(
                f"Cannot store empty credential for {provider}:{credential_type}."
            )

        try:
            key = self._key(provider, credential_type)
            encrypted_value = self._encrypt_credential(value)
            keyring.set_password(
                self.service_name,
                key,
                encrypted_value,
            )
        except CredentialError:
            raise
        except Exception as exc:
            raise CredentialError(
                f"Failed to store credential for {provider}: {exc}"
            ) from exc

    def get_credential(
        self,
        provider: str,
        credential_type: str,
    ) -> Optional[str]:
        try:
            key = self._key(provider, credential_type)
            encrypted_value = keyring.get_password(
                self.service_name,
                key,
            )

            if not encrypted_value:
                return None

            return self._decrypt_credential(encrypted_value)
        except CredentialError:
            raise
        except Exception as exc:
            raise CredentialError(
                f"Failed to retrieve credential for {provider}: {exc}"
            ) from exc

    def delete_credential(
        self,
        provider: str,
        credential_type: str,
    ) -> None:
        try:
            key = self._key(provider, credential_type)

            try:
                keyring.delete_password(
                    self.service_name,
                    key,
                )
            except keyring.errors.PasswordDeleteError:
                return
        except CredentialError:
            raise
        except Exception as exc:
            raise CredentialError(
                f"Failed to delete credential for {provider}: {exc}"
            ) from exc

    def list_credentials(self) -> Dict[str, Dict[str, str]]:
        credential_types = {
            "aws": [
                "access_key_id",
                "secret_access_key",
                "session_token",
            ],
            "azure": [
                "tenant_id",
                "client_id",
                "client_secret",
                "subscription_id",
            ],
            "gcp": [
                "service_account_key",
            ],
        }

        result: Dict[str, Dict[str, str]] = {
            provider: {} for provider in credential_types
        }

        for provider, types in credential_types.items():
            for credential_type in types:
                try:
                    value = self.get_credential(provider, credential_type)
                    if value:
                        result[provider][credential_type] = "stored"
                except CredentialError:
                    continue

        return result

    def setup_aws_credentials(
        self,
        access_key_id: str,
        secret_access_key: str,
        session_token: Optional[str] = None,
    ) -> None:
        self.store_credential(
            "aws",
            "access_key_id",
            access_key_id,
        )
        self.store_credential(
            "aws",
            "secret_access_key",
            secret_access_key,
        )

        if session_token:
            self.store_credential(
                "aws",
                "session_token",
                session_token,
            )

    def setup_azure_credentials(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        subscription_id: str,
    ) -> None:
        self.store_credential(
            "azure",
            "tenant_id",
            tenant_id,
        )
        self.store_credential(
            "azure",
            "client_id",
            client_id,
        )
        self.store_credential(
            "azure",
            "client_secret",
            client_secret,
        )
        self.store_credential(
            "azure",
            "subscription_id",
            subscription_id,
        )

    def setup_gcp_credentials(
        self,
        service_account_key: str,
    ) -> None:
        if isinstance(service_account_key, dict):
            service_account_key = json.dumps(service_account_key)

        self.store_credential(
            "gcp",
            "service_account_key",
            service_account_key,
        )

    def get_aws_credentials(self) -> Dict[str, str]:
        credentials: Dict[str, str] = {}

        mapping = {
            "access_key_id": "AWS_ACCESS_KEY_ID",
            "secret_access_key": "AWS_SECRET_ACCESS_KEY",
            "session_token": "AWS_SESSION_TOKEN",
        }

        for credential_type, environment_key in mapping.items():
            value = self.get_credential(
                "aws",
                credential_type,
            )

            if not value:
                value = os.getenv(environment_key)

            if value:
                credentials[credential_type] = value

        return credentials

    def get_azure_credentials(self) -> Dict[str, str]:
        credentials: Dict[str, str] = {}

        mapping = {
            "tenant_id": "AZURE_TENANT_ID",
            "client_id": "AZURE_CLIENT_ID",
            "client_secret": "AZURE_CLIENT_SECRET",
            "subscription_id": "AZURE_SUBSCRIPTION_ID",
        }

        for credential_type, environment_key in mapping.items():
            value = self.get_credential(
                "azure",
                credential_type,
            )

            if not value:
                value = os.getenv(environment_key)

            if value:
                credentials[credential_type] = value

        return credentials

    def get_gcp_credentials(self) -> Dict[str, str]:
        credentials: Dict[str, str] = {}

        value = self.get_credential(
            "gcp",
            "service_account_key",
        )

        if not value:
            environment_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

            if environment_key and Path(environment_key).exists():
                value = Path(environment_key).read_text(
                    encoding="utf-8"
                )

            elif os.getenv("GCP_SERVICE_ACCOUNT_KEY"):
                value = os.getenv("GCP_SERVICE_ACCOUNT_KEY")

        if value:
            credentials["service_account_key"] = value

        return credentials

    def validate_credentials(self, provider: str) -> bool:
        provider = provider.strip().lower()

        try:
            if provider == "aws":
                credentials = self.get_aws_credentials()
                return bool(
                    credentials.get("access_key_id")
                    and credentials.get("secret_access_key")
                )

            if provider == "azure":
                credentials = self.get_azure_credentials()
                return all(
                    credentials.get(key)
                    for key in (
                        "tenant_id",
                        "client_id",
                        "client_secret",
                        "subscription_id",
                    )
                )

            if provider == "gcp":
                credentials = self.get_gcp_credentials()
                return bool(credentials.get("service_account_key"))

        except CredentialError:
            return False

        return False

    def clear_all_credentials(self) -> None:
        credentials = {
            "aws": [
                "access_key_id",
                "secret_access_key",
                "session_token",
            ],
            "azure": [
                "tenant_id",
                "client_id",
                "client_secret",
                "subscription_id",
            ],
            "gcp": [
                "service_account_key",
            ],
        }

        for provider, credential_types in credentials.items():
            for credential_type in credential_types:
                self.delete_credential(
                    provider,
                    credential_type,
                )