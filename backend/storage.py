import os
import shutil
from abc import ABC, abstractmethod

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "storage"))

class IStorageProvider(ABC):
    @abstractmethod
    def save_file(self, local_source: str, relative_dest: str) -> str:
        """Saves a file to storage and returns its address/path."""
        pass
        
    @abstractmethod
    def get_file_url(self, relative_path: str) -> str:
        """Returns accessibility URL/URI path for the asset."""
        pass

class LocalDiskStorageProvider(IStorageProvider):
    def __init__(self, root_dir: str = STORAGE_ROOT):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        
    def save_file(self, local_source: str, relative_dest: str) -> str:
        dest_abs = os.path.join(self.root_dir, relative_dest)
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        shutil.copy(local_source, dest_abs)
        return relative_dest # return relative path as the identifier
        
    def get_file_url(self, relative_path: str) -> str:
        # Standard relative mounting endpoint for local web assets
        # Standardizes forward slashes for cross-platform browser support
        return "/" + relative_path.replace("\\", "/")

class S3StorageProvider(IStorageProvider):
    def __init__(self, bucket_name: str, endpoint_url: str = None, aws_key: str = None, aws_secret: str = None):
        self.bucket = bucket_name
        self.client = None
        self.loaded = False
        try:
            import boto3
            # Set up MinIO or real AWS S3 client
            self.client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret
            )
            self.loaded = True
        except ImportError:
            print("[Warning] boto3 not installed. S3StorageProvider running in local fallback.")
            
    def save_file(self, local_source: str, relative_dest: str) -> str:
        if self.loaded and self.client:
            try:
                # S3 upload
                key = relative_dest.replace("\\", "/")
                self.client.upload_file(local_source, self.bucket, key)
                return f"s3://{self.bucket}/{key}"
            except Exception as e:
                print(f"S3 Upload Error, falling back: {e}")
        
        # Fallback to local disk copy if S3 fails or boto3 not present
        dest_abs = os.path.join(STORAGE_ROOT, relative_dest)
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        shutil.copy(local_source, dest_abs)
        return relative_dest
        
    def get_file_url(self, relative_path: str) -> str:
        if self.loaded and self.client:
            # Return S3 URL (could generate pre-signed URL or public URL)
            clean_path = relative_path.replace('\\', '/')
            return f"https://s3.amazonaws.com/{self.bucket}/{clean_path}"
        return "/" + relative_path.replace("\\", "/")

# Factory manager
class StorageProvider:
    _instance = None
    
    @classmethod
    def get_provider(cls) -> IStorageProvider:
        if cls._instance is None:
            # Check env settings
            storage_type = os.getenv("STORAGE_TYPE", "LOCAL").upper()
            if storage_type == "S3":
                cls._instance = S3StorageProvider(
                    bucket_name=os.getenv("S3_BUCKET", "diagrams-v2"),
                    endpoint_url=os.getenv("S3_ENDPOINT", None), # MinIO url e.g. http://localhost:9000
                    aws_key=os.getenv("AWS_ACCESS_KEY_ID", None),
                    aws_secret=os.getenv("AWS_SECRET_ACCESS_KEY", None)
                )
            else:
                cls._instance = LocalDiskStorageProvider()
        return cls._instance
