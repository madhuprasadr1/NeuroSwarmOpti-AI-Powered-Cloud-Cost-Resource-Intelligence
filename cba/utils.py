import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .core.exceptions import CloudBillingError


class EncryptionUtils:
    def __init__(self, key_file: Optional[Path] = None):
        self.key_file = Path(
            key_file or Path.home() / ".cba" / "encryption.key"
        ).expanduser()
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self._encryption_key: Optional[bytes] = None
        self._fernet: Optional[Fernet] = None

    def get_or_create_key(self) -> bytes:
        if self._encryption_key is not None:
            return self._encryption_key

        try:
            if self.key_file.exists():
                key_data = self.key_file.read_bytes()
                Fernet(key_data)
                self._encryption_key = key_data
                return key_data

            return self._create_new_key()

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to load encryption key: {exc}"
            ) from exc

    def _create_new_key(self) -> bytes:
        key = Fernet.generate_key()

        try:
            self.key_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.key_file.write_bytes(key)

            if os.name != "nt":
                os.chmod(self.key_file, 0o600)

            self._encryption_key = key
            self._fernet = Fernet(key)

            return key

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to create encryption key: {exc}"
            ) from exc

    def get_fernet(self) -> Fernet:
        if self._fernet is None:
            self._fernet = Fernet(
                self.get_or_create_key()
            )

        return self._fernet

    def encrypt_data(self, data: str) -> str:
        if not isinstance(data, str):
            raise CloudBillingError(
                "Data to encrypt must be a string."
            )

        try:
            return self.get_fernet().encrypt(
                data.encode("utf-8")
            ).decode("utf-8")
        except Exception as exc:
            raise CloudBillingError(
                f"Failed to encrypt data: {exc}"
            ) from exc

    def decrypt_data(self, encrypted_data: str) -> str:
        if not isinstance(encrypted_data, str):
            raise CloudBillingError(
                "Encrypted data must be a string."
            )

        try:
            return self.get_fernet().decrypt(
                encrypted_data.encode("utf-8")
            ).decode("utf-8")
        except Exception as exc:
            raise CloudBillingError(
                f"Failed to decrypt data: {exc}"
            ) from exc

    def encrypt_dict(
        self,
        data: Dict[str, Any],
    ) -> str:
        try:
            serialized = json.dumps(
                data,
                separators=(",", ":"),
                default=str,
                ensure_ascii=False,
            )
            return self.encrypt_data(serialized)
        except CloudBillingError:
            raise
        except Exception as exc:
            raise CloudBillingError(
                f"Failed to encrypt dictionary: {exc}"
            ) from exc

    def decrypt_dict(
        self,
        encrypted_data: str,
    ) -> Dict[str, Any]:
        try:
            decrypted = self.decrypt_data(encrypted_data)
            result = json.loads(decrypted)

            if not isinstance(result, dict):
                raise CloudBillingError(
                    "Decrypted value is not a dictionary."
                )

            return result

        except CloudBillingError:
            raise
        except Exception as exc:
            raise CloudBillingError(
                f"Failed to decrypt dictionary: {exc}"
            ) from exc

    def encrypt_file(
        self,
        file_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        file_path = Path(file_path).expanduser()

        if not file_path.exists():
            raise CloudBillingError(
                f"File not found: {file_path}"
            )

        if not file_path.is_file():
            raise CloudBillingError(
                f"Path is not a file: {file_path}"
            )

        if output_path is None:
            output_path = file_path.with_suffix(
                file_path.suffix + ".enc"
            )
        else:
            output_path = Path(output_path).expanduser()

        try:
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            encrypted_data = self.get_fernet().encrypt(
                file_path.read_bytes()
            )

            output_path.write_bytes(encrypted_data)

            if os.name != "nt":
                os.chmod(output_path, 0o600)

            return output_path

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to encrypt file {file_path}: {exc}"
            ) from exc

    def decrypt_file(
        self,
        encrypted_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        encrypted_path = Path(
            encrypted_path
        ).expanduser()

        if not encrypted_path.exists():
            raise CloudBillingError(
                f"Encrypted file not found: {encrypted_path}"
            )

        if output_path is None:
            if encrypted_path.suffix == ".enc":
                output_path = encrypted_path.with_suffix("")
            else:
                output_path = encrypted_path.with_suffix(".dec")
        else:
            output_path = Path(output_path).expanduser()

        try:
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            decrypted_data = self.get_fernet().decrypt(
                encrypted_path.read_bytes()
            )

            output_path.write_bytes(decrypted_data)

            if os.name != "nt":
                os.chmod(output_path, 0o600)

            return output_path

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to decrypt file {encrypted_path}: {exc}"
            ) from exc

    def hash_data(
        self,
        data: Any,
        algorithm: str = "sha256",
    ) -> str:
        algorithm = str(algorithm).lower()

        algorithms = {
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
            "md5": hashlib.md5,
        }

        if algorithm not in algorithms:
            raise CloudBillingError(
                f"Unsupported hash algorithm: {algorithm}"
            )

        if isinstance(data, bytes):
            payload = data
        elif isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = str(data).encode("utf-8")

        try:
            return algorithms[algorithm](payload).hexdigest()
        except Exception as exc:
            raise CloudBillingError(
                f"Failed to hash data: {exc}"
            ) from exc

    def generate_secure_token(
        self,
        length: int = 32,
    ) -> str:
        if length <= 0:
            raise CloudBillingError(
                "Token length must be greater than zero."
            )

        return secrets.token_urlsafe(length)

    def generate_api_key(
        self,
        prefix: str = "cba",
    ) -> str:
        prefix = str(prefix).strip()

        if not prefix:
            prefix = "cba"

        return f"{prefix}_{secrets.token_urlsafe(32)}"

    def derive_key_from_password(
        self,
        password: str,
        salt: Optional[bytes] = None,
        iterations: int = 100000,
    ) -> bytes:
        if not isinstance(password, str) or not password:
            raise CloudBillingError(
                "Password must be a non-empty string."
            )

        if iterations < 10000:
            raise CloudBillingError(
                "PBKDF2 iterations must be at least 10000."
            )

        if salt is None:
            salt = secrets.token_bytes(32)

        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
            )

            return kdf.derive(
                password.encode("utf-8")
            )

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to derive encryption key: {exc}"
            ) from exc

    def encrypt_with_password(
        self,
        data: str,
        password: str,
        salt: Optional[bytes] = None,
    ) -> tuple[str, bytes]:
        if salt is None:
            salt = secrets.token_bytes(32)

        key = self.derive_key_from_password(
            password,
            salt,
        )

        try:
            encrypted_data = Fernet(key).encrypt(
                data.encode("utf-8")
            )

            return (
                encrypted_data.decode("utf-8"),
                salt,
            )

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to encrypt with password: {exc}"
            ) from exc

    def decrypt_with_password(
        self,
        encrypted_data: str,
        password: str,
        salt: bytes,
    ) -> str:
        if not salt:
            raise CloudBillingError(
                "Salt is required for password decryption."
            )

        key = self.derive_key_from_password(
            password,
            salt,
        )

        try:
            return Fernet(key).decrypt(
                encrypted_data.encode("utf-8")
            ).decode("utf-8")

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to decrypt with password: {exc}"
            ) from exc

    def secure_delete_file(
        self,
        file_path: Path,
        passes: int = 3,
    ) -> None:
        file_path = Path(file_path).expanduser()

        if not file_path.exists():
            return

        if passes <= 0:
            raise CloudBillingError(
                "passes must be greater than zero."
            )

        try:
            file_size = file_path.stat().st_size

            with file_path.open("r+b") as file:
                for _ in range(passes):
                    file.seek(0)
                    remaining = file_size

                    while remaining > 0:
                        chunk_size = min(
                            remaining,
                            1024 * 1024,
                        )
                        file.write(
                            secrets.token_bytes(chunk_size)
                        )
                        remaining -= chunk_size

                    file.flush()

                    try:
                        os.fsync(file.fileno())
                    except OSError:
                        pass

            file_path.unlink()

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to securely delete file {file_path}: {exc}"
            ) from exc

    def verify_file_integrity(
        self,
        file_path: Path,
        expected_hash: str,
        algorithm: str = "sha256",
    ) -> bool:
        file_path = Path(file_path).expanduser()

        if not file_path.exists():
            return False

        try:
            actual_hash = self.create_file_hash(
                file_path,
                algorithm,
            )

            return secrets.compare_digest(
                actual_hash.lower(),
                str(expected_hash).lower(),
            )

        except Exception:
            return False

    def create_file_hash(
        self,
        file_path: Path,
        algorithm: str = "sha256",
    ) -> str:
        file_path = Path(file_path).expanduser()

        if not file_path.exists():
            raise CloudBillingError(
                f"File not found: {file_path}"
            )

        if not file_path.is_file():
            raise CloudBillingError(
                f"Path is not a file: {file_path}"
            )

        try:
            hash_algorithm = str(algorithm).lower()

            if hash_algorithm == "sha256":
                digest = hashlib.sha256()
            elif hash_algorithm == "sha512":
                digest = hashlib.sha512()
            elif hash_algorithm == "md5":
                digest = hashlib.md5()
            else:
                raise CloudBillingError(
                    f"Unsupported hash algorithm: {algorithm}"
                )

            with file_path.open("rb") as file:
                while True:
                    chunk = file.read(1024 * 1024)

                    if not chunk:
                        break

                    digest.update(chunk)

            return digest.hexdigest()

        except CloudBillingError:
            raise
        except Exception as exc:
            raise CloudBillingError(
                f"Failed to create file hash: {exc}"
            ) from exc

    def encrypt_sensitive_dict(
        self,
        data: Dict[str, Any],
        sensitive_keys: List[str],
    ) -> Dict[str, Any]:
        encrypted_data = dict(data)

        for key in sensitive_keys:
            if (
                key in encrypted_data
                and encrypted_data[key] is not None
            ):
                value = encrypted_data[key]

                if isinstance(value, str):
                    encrypted_data[key] = self.encrypt_data(
                        value
                    )
                else:
                    encrypted_data[key] = self.encrypt_data(
                        json.dumps(
                            value,
                            default=str,
                        )
                    )

        return encrypted_data

    def decrypt_sensitive_dict(
        self,
        data: Dict[str, Any],
        sensitive_keys: List[str],
    ) -> Dict[str, Any]:
        decrypted_data = dict(data)

        for key in sensitive_keys:
            if (
                key in decrypted_data
                and isinstance(decrypted_data[key], str)
            ):
                try:
                    value = self.decrypt_data(
                        decrypted_data[key]
                    )

                    try:
                        decrypted_data[key] = json.loads(
                            value
                        )
                    except json.JSONDecodeError:
                        decrypted_data[key] = value

                except CloudBillingError:
                    continue

        return decrypted_data

    def rotate_encryption_key(self) -> str:
        backup_path: Optional[Path] = None

        try:
            if self.key_file.exists():
                backup_path = self.key_file.with_suffix(
                    ".backup"
                )

                if backup_path.exists():
                    backup_path.unlink()

                self.key_file.replace(backup_path)

            self._encryption_key = None
            self._fernet = None

            new_key = self._create_new_key()

            if backup_path:
                return (
                    f"New encryption key created. "
                    f"Previous key backed up to {backup_path}"
                )

            return "New encryption key created."

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to rotate encryption key: {exc}"
            ) from exc

    def get_key_info(self) -> Dict[str, Any]:
        try:
            if not self.key_file.exists():
                return {
                    "key_file": str(self.key_file),
                    "exists": False,
                }

            stat = self.key_file.stat()

            return {
                "key_file": str(self.key_file),
                "exists": True,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "permissions": (
                    oct(stat.st_mode)[-3:]
                    if os.name != "nt"
                    else None
                ),
            }

        except Exception as exc:
            raise CloudBillingError(
                f"Failed to get key info: {exc}"
            ) from exc