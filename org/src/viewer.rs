use std::convert::TryInto;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Opt {
    input: PathBuf,
    skip_bytes: u64,
}

impl Opt {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: binary-parser <input> <skip_bytes>");
            std::process::exit(1);
        }

        Self {
            input: PathBuf::from(&args[1]),
            skip_bytes: args.get(2).and_then(|arg| arg.parse().ok()).unwrap_or(0),
        }
    }
}

#[allow(unused)]
#[derive(Debug)]
enum Distance {
    CosineSimilarity(f32),
    CosineDistance(f32),
    EuclideanDistance(f32),
    HammingDistance(f32),
    DotProductDistance(f32),
    Invalid(u8),
}

#[derive(Debug)]
struct IndexPtr {
    offset: u32,
    file_id: u32,
}

#[derive(Debug)]
struct LinkValidation {
    is_valid: bool,
    reason: Option<String>,
}

impl IndexPtr {
    fn read(reader: &mut impl Read) -> io::Result<Self> {
        let mut buffer = [0u8; 8];
        reader.read_exact(&mut buffer)?;

        Ok(IndexPtr {
            offset: u32::from_le_bytes(buffer[0..4].try_into().unwrap()),
            file_id: u32::from_le_bytes(buffer[4..8].try_into().unwrap()),
        })
    }

    fn is_absent(&self) -> bool {
        self.offset == u32::MAX || self.file_id == u32::MAX
    }

    fn validate_link(&self, dir: &Path) -> LinkValidation {
        if self.is_absent() {
            return LinkValidation {
                is_valid: true,
                reason: None,
            };
        }

        let index_file = dir.join(format!("{}.index", self.file_id));

        if !index_file.exists() {
            return LinkValidation {
                is_valid: false,
                reason: Some(format!("Index file {:?} not found", index_file)),
            };
        }

        let file_size = std::fs::metadata(&index_file).map(|m| m.len()).unwrap_or(0);

        if (self.offset as u64) >= file_size {
            return LinkValidation {
                is_valid: false,
                reason: Some(format!(
                    "Offset {} exceeds file size {}",
                    self.offset, file_size
                )),
            };
        }

        LinkValidation {
            is_valid: true,
            reason: None,
        }
    }
}

#[derive(Debug)]
struct Neighbor {
    id: u32,
    ptr: IndexPtr,
    distance: Distance,
    validation: LinkValidation,
}

#[derive(Debug)]
struct Node {
    level: u8,
    version: u32,
    prop_offset: u32,
    prop_length: u32,
    metadata_offset: u32,
    metadata_length: u32,
    parent_ptr: IndexPtr,
    parent_validation: LinkValidation,
    child_ptr: IndexPtr,
    child_validation: LinkValidation,
    neighbors: Vec<Option<Neighbor>>,
}

fn resolve_latest_ptr(latest_version_file: &mut File, offset: u32) -> io::Result<IndexPtr> {
    if offset == u32::MAX {
        return Ok(IndexPtr {
            offset: u32::MAX,
            file_id: u32::MAX,
        });
    }
    latest_version_file.seek(SeekFrom::Start(offset as u64))?;
    IndexPtr::read(latest_version_file)
}

fn read_neighbors(
    reader: &mut impl Read,
    latest_version_file: &mut File,
    dir: &Path,
) -> io::Result<Vec<Option<Neighbor>>> {
    let mut length = [0u8; 2];
    reader.read_exact(&mut length)?;
    let length = u16::from_le_bytes(length);

    let mut neighbors = Vec::with_capacity(length as usize);
    for _ in 0..length {
        let mut id_buffer = [0u8; 4];
        reader.read_exact(&mut id_buffer)?;
        let id = u32::from_le_bytes(id_buffer);

        if id == u32::MAX {
            let mut skip_buffer = [0u8; 9];
            reader.read_exact(&mut skip_buffer)?;
            neighbors.push(None);
            continue;
        }

        let mut latest_offset_buffer = [0u8; 4];
        reader.read_exact(&mut latest_offset_buffer)?;
        let latest_offset = u32::from_le_bytes(latest_offset_buffer);

        let ptr = resolve_latest_ptr(latest_version_file, latest_offset)?;
        let validation = ptr.validate_link(dir);

        let mut distance_type = [0u8; 1];
        reader.read_exact(&mut distance_type)?;

        let mut distance_value = [0u8; 4];
        reader.read_exact(&mut distance_value)?;
        let value = f32::from_le_bytes(distance_value);

        let distance = match distance_type[0] {
            0 => Distance::CosineSimilarity(value),
            1 => Distance::CosineDistance(value),
            2 => Distance::EuclideanDistance(value),
            3 => Distance::HammingDistance(value),
            4 => Distance::DotProductDistance(value),
            n => Distance::Invalid(n),
        };

        neighbors.push(Some(Neighbor {
            id,
            ptr,
            distance,
            validation,
        }));
    }

    Ok(neighbors)
}

fn read_node(file: &mut File, latest_version_file: &mut File, dir: &Path) -> io::Result<Node> {
    let mut level = [0u8; 1];
    file.read_exact(&mut level)?;
    let level = level[0];

    let mut version_buffer = [0u8; 4];
    file.read_exact(&mut version_buffer)?;
    let version = u32::from_le_bytes(version_buffer);

    let mut prop_buffer = [0u8; 8];
    file.read_exact(&mut prop_buffer)?;
    let prop_offset = u32::from_le_bytes(prop_buffer[0..4].try_into().unwrap());
    let prop_length = u32::from_le_bytes(prop_buffer[4..8].try_into().unwrap());

    let mut metadata_buffer = [0u8; 8];
    file.read_exact(&mut metadata_buffer)?;
    let metadata_offset = u32::from_le_bytes(metadata_buffer[0..4].try_into().unwrap());
    let metadata_length = u32::from_le_bytes(metadata_buffer[4..8].try_into().unwrap());

    let mut parent_latest_buffer = [0u8; 4];
    file.read_exact(&mut parent_latest_buffer)?;
    let parent_latest_offset = u32::from_le_bytes(parent_latest_buffer);
    let parent_ptr = resolve_latest_ptr(latest_version_file, parent_latest_offset)?;
    let parent_validation = parent_ptr.validate_link(dir);

    let mut child_latest_buffer = [0u8; 4];
    file.read_exact(&mut child_latest_buffer)?;
    let child_latest_offset = u32::from_le_bytes(child_latest_buffer);
    let child_ptr = resolve_latest_ptr(latest_version_file, child_latest_offset)?;
    let child_validation = child_ptr.validate_link(dir);

    let neighbors = read_neighbors(file, latest_version_file, dir)?;

    Ok(Node {
        level,
        version,
        prop_offset,
        prop_length,
        metadata_offset,
        metadata_length,
        parent_ptr,
        parent_validation,
        child_ptr,
        child_validation,
        neighbors,
    })
}

fn print_index_ptr(ptr: &IndexPtr, validation: &LinkValidation, prefix: &str) {
    if ptr.is_absent() {
        println!("{}: absent", prefix);
        return;
    }

    print!("{}: offset={}, file_id={}", prefix, ptr.offset, ptr.file_id);

    if !validation.is_valid {
        if let Some(reason) = &validation.reason {
            print!(" (INVALID: {})", reason);
        }
    }
    println!();
}

fn print_node(node: &Node) {
    println!("HNSW Level: {}", node.level);
    println!("Version: {}", node.version);
    println!(
        "Prop Offset: {}, Length: {}",
        node.prop_offset, node.prop_length
    );

    if node.metadata_offset != u32::MAX {
        println!(
            "Metadata Offset: {}, Length: {}",
            node.metadata_offset, node.metadata_length
        );
    } else {
        println!("Metadata: absent");
    }

    print_index_ptr(&node.parent_ptr, &node.parent_validation, "Parent");
    print_index_ptr(&node.child_ptr, &node.child_validation, "Child");

    println!("\nNeighbors:");
    for (i, neighbor) in node.neighbors.iter().enumerate() {
        match neighbor {
            Some(n) => {
                print!(
                    "  {}. ID: {}, Version: offset={}, file_id={}, Distance: {:?}",
                    i + 1,
                    n.id,
                    n.ptr.offset,
                    n.ptr.file_id,
                    n.distance
                );
                if !n.validation.is_valid {
                    if let Some(reason) = &n.validation.reason {
                        print!(" (INVALID: {})", reason);
                    }
                }
                println!();
            }
            None => {
                println!("  {}. ABSENT", i + 1);
            }
        }
    }
}

fn main() -> io::Result<()> {
    let opt = Opt::from_args();
    let mut file = File::open(&opt.input)?;
    let dir = opt.input.parent().unwrap_or_else(|| Path::new("."));
    let latest_version_path = dir.join("latest.version");
    let mut latest_version_file = File::open(latest_version_path)?;

    if opt.skip_bytes > 0 {
        file.seek(SeekFrom::Start(opt.skip_bytes))?;
    }

    let node = read_node(&mut file, &mut latest_version_file, dir)?;
    print_node(&node);

    Ok(())
}
