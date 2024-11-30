
use aws_sdk_s3::Client;
use aws_types::region::Region;
use aws_credential_types::Credentials;
use aws_sdk_s3::primitives::ByteStream;
use std::fs::File;
use std::io::Read;
use chrono::Utc;
use std::error::Error;

struct S3Client {
    client: Client,
    bucket: String,
}

impl S3Client {
    fn new(
        access_key: String,
        secret_key: String,
        region: String,
        bucket: String,
    ) -> Result<Self, Box<dyn Error>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        
        rt.block_on(async {
            // Create credentials
            let credentials = Credentials::new(
                access_key,
                secret_key,
                None,
                None,
                "static",
            );

            // Create config
            let config = aws_config::from_env()
                .region(Region::new(region))
                .credentials_provider(credentials)
                .load()
                .await;

            // Create client
            let client = Client::new(&config);

            Ok(S3Client {
                client,
                bucket,
            })
        })
    }

    fn generate_key(&self, prefix: &str, filename: &str) -> String {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        format!("{}/{}__{}", prefix, timestamp, filename)
    }

    fn upload_file(&self, file_path: &str, prefix: &str) -> Result<String, Box<dyn Error>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        
        rt.block_on(async {
            let mut file = File::open(file_path)?;
            let mut contents = Vec::new();
            file.read_to_end(&mut contents)?;

            let filename = std::path::Path::new(file_path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap();
            let key = self.generate_key(prefix, filename);

            self.client
                .put_object()
                .bucket(&self.bucket)
                .key(&key)
                .body(ByteStream::from(contents))
                .send()
                .await?;

            println!("Uploaded file to s3://{}/{}", self.bucket, key);
            Ok(key)
        })
    }

    fn download_file(&self, key: &str, output_path: &str) -> Result<(), Box<dyn Error>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        
        rt.block_on(async {
            let result = self.client
                .get_object()
                .bucket(&self.bucket)
                .key(key)
                .send()
                .await?;

            let bytes = result.body.collect().await?.into_bytes();
            std::fs::write(output_path, bytes)?;

            println!("Downloaded s3://{}/{} to {}", self.bucket, key, output_path);
            Ok(())
        })
    }

    fn list_objects(&self, prefix: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        
        rt.block_on(async {
            let result = self.client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(prefix)
                .send()
                .await?;

            let mut keys = Vec::new();
            if let Some(objects) = result.contents {
                for obj in objects {
                    if let Some(key) = obj.key {
                        keys.push(key);
                    }
                }
            }

            Ok(keys)
        })
    }

    fn delete_object(&self, key: &str) -> Result<(), Box<dyn Error>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        
        rt.block_on(async {
            self.client
                .delete_object()
                .bucket(&self.bucket)
                .key(key)
                .send()
                .await?;

            println!("Deleted s3://{}/{}", self.bucket, key);
            Ok(())
        })
    }
}

// More efficient version using a shared runtime
struct S3ClientEfficient {
    client: Client,
    bucket: String,
    runtime: tokio::runtime::Runtime,
}

impl S3ClientEfficient {
    fn new(
        access_key: String,
        secret_key: String,
        region: String,
        bucket: String,
    ) -> Result<Self, Box<dyn Error>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        
        let client = runtime.block_on(async {
            let credentials = Credentials::new(
                access_key,
                secret_key,
                None,
                None,
                "static",
            );

            let config = aws_config::from_env()
                .region(Region::new(region))
                .credentials_provider(credentials)
                .load()
                .await;

            Client::new(&config)
        });

        Ok(S3ClientEfficient {
            client,
            bucket,
            runtime,
        })
    }

    fn upload_file(&self, file_path: &str, prefix: &str) -> Result<String, Box<dyn Error>> {
        self.runtime.block_on(async {
            // Same upload implementation as before
            let mut file = File::open(file_path)?;
            let mut contents = Vec::new();
            file.read_to_end(&mut contents)?;

            let filename = std::path::Path::new(file_path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap();
            let key = format!("{}/{}", prefix, filename);

            self.client
                .put_object()
                .bucket(&self.bucket)
                .key(&key)
                .body(ByteStream::from(contents))
                .send()
                .await?;

            Ok(key)
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize client
    let s3_client = S3Client::new(
        "AKIA2CUNLIWONNM4DO66".to_string(),
        "---FILL HERE---".to_string(),
        "eu-north-1".to_string(),
        "cosdata-channi--eun1-az2--x-s3".to_string(),
    )?;

    // Upload example
    let upload_key = s3_client.upload_file("local_file.txt", "uploads")?;

    // List objects
    let objects = s3_client.list_objects("uploads/")?;
    println!("Objects in bucket:");
    for obj in objects {
        println!("- {}", obj);
    }

    // Download example
    s3_client.download_file(&upload_key, "downloaded_file.txt")?;

    // Delete example
    //s3_client.delete_object(&upload_key)?;

    Ok(())
}
