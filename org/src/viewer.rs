use std::convert::TryInto;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Opt {
    input: PathBuf,
    skip_bytes: u64,
    limit: usize,
}

impl Opt {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 3 {
            eprintln!("Usage: binary-parser <input> <skip_bytes> <limit>");
            std::process::exit(1);
        }

        Self {
            input: PathBuf::from(&args[1]),
            skip_bytes: args[2].parse().unwrap_or(0),
            limit: args.get(3).map(|s| s.parse().unwrap_or(1)).unwrap_or(1),
        }
    }
}

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
        self.offset == u32::MAX
    }

    fn validate_link(&self, current_file: &Path, current_file_id: u32) -> LinkValidation {
        if self.is_absent() {
            return LinkValidation {
                is_valid: true,
                reason: None,
            };
        }

        // Get current file's directory and version
        let dir = current_file.parent().unwrap_or_else(|| Path::new("."));

        // If same hash, just check current file
        if self.file_id == current_file_id {
            let file_size = std::fs::metadata(current_file)
                .map(|m| m.len())
                .unwrap_or(0);

            if (self.offset as u64) >= file_size {
                return LinkValidation {
                    is_valid: false,
                    reason: Some(format!(
                        "Offset {} exceeds file size {}",
                        self.offset, file_size
                    )),
                };
            }
            return LinkValidation {
                is_valid: true,
                reason: None,
            };
        }

        // Check for version file existence using hash
        let index_file = dir.join(format!("{}.index", self.file_id));

        if !index_file.exists() {
            return LinkValidation {
                is_valid: false,
                reason: Some(format!("Index file {:?} not found", index_file)),
            };
        }

        // Check file size
        let file_size = std::fs::metadata(&index_file).map(|m| m.len()).unwrap_or(0);

        if (self.offset as u64) >= file_size {
            return LinkValidation {
                is_valid: false,
                reason: Some(format!(
                    "Offset {} exceeds file size {} in {:?}",
                    self.offset, file_size, index_file
                )),
            };
        }

        LinkValidation {
            is_valid: true,
            reason: None,
        }
    }
}

static mut LEVEL0_NEIGHBORS: u16 = 0;
static mut LEVELN_NEIGHBORS: u16 = 0;

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
    version: u64,
    prop_offset: u32,
    prop_length: u32,
    metadata_offset: u32,
    metadata_length: u32,
    parent: IndexPtr,
    parent_validation: LinkValidation,
    child: IndexPtr,
    child_validation: LinkValidation,
    root_version: IndexPtr,
    root_version_validation: LinkValidation,
    root_version_is_root: bool,
    neighbors: Vec<Option<Neighbor>>,
}

fn read_neighbors(
    reader: &mut impl Read,
    current_file: &Path,
    current_file_id: u32,
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
            let mut skip_buffer = [0u8; 13];
            reader.read_exact(&mut skip_buffer)?;
            neighbors.push(None);
            continue;
        }

        let ptr = IndexPtr::read(reader)?;
        let validation = ptr.validate_link(current_file, current_file_id);

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

fn read_node(file: &mut File, current_file: &Path, current_file_id: u32) -> io::Result<Node> {
    let mut level = [0u8; 1];
    file.read_exact(&mut level)?;
    let level = level[0];
    let mut version_hash = [0u8; 8];

    file.read_exact(&mut version_hash)?;
    let version_hash = u64::from_le_bytes(version_hash);

    // Read property offset and length (8 bytes)
    let mut buffer = [0u8; 8];
    file.read_exact(&mut buffer)?;
    let prop_offset = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
    let prop_length = u32::from_le_bytes(buffer[4..8].try_into().unwrap());

    // Read metadata offset and length (8 bytes) - new in updated serialization
    let mut metadata_buffer = [0u8; 8];
    file.read_exact(&mut metadata_buffer)?;
    let metadata_offset = u32::from_le_bytes(metadata_buffer[0..4].try_into().unwrap());
    let metadata_length = u32::from_le_bytes(metadata_buffer[4..8].try_into().unwrap());

    let parent = IndexPtr::read(file)?;
    let parent_validation = parent.validate_link(current_file, current_file_id);

    let child = IndexPtr::read(file)?;
    let child_validation = child.validate_link(current_file, current_file_id);

    let mut root_version_is_root = [0u8; 1];
    file.read_exact(&mut root_version_is_root)?;
    let root_version_is_root = root_version_is_root[0] != 0;
    let root_version = IndexPtr::read(file)?;

    let root_version_validation = root_version.validate_link(current_file, current_file_id);

    let neighbors = read_neighbors(file, current_file, current_file_id)?;

    Ok(Node {
        level,
        version: version_hash,
        prop_offset,
        prop_length,
        metadata_offset,
        metadata_length,
        parent,
        parent_validation,
        child,
        child_validation,
        root_version,
        root_version_validation,
        root_version_is_root,
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

fn print_node(node: &Node, index: usize) {
    println!("\nNode {}:", index);
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

    print_index_ptr(&node.parent, &node.parent_validation, "Parent");
    print_index_ptr(&node.child, &node.child_validation, "Child");

    // Display the root version with its tag info
    if node.root_version.is_absent() {
        println!("Root version: absent");
    } else {
        print!(
            "Root version: offset={}, file_id={}, is_root={}",
            node.root_version.offset, node.root_version.file_id, node.root_version_is_root
        );
        if !node.root_version_validation.is_valid {
            if let Some(reason) = &node.root_version_validation.reason {
                print!(" (INVALID: {})", reason);
            }
        }
        println!();
    }

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

    // Extract version hash from filename
    let current_file_id = opt
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.parse::<u32>().ok())
        .expect("Invalid file name");

    if opt.skip_bytes > 0 {
        file.seek(SeekFrom::Start(opt.skip_bytes))?;
    }

    for i in 0..opt.limit {
        match read_node(&mut file, &opt.input, current_file_id) {
            Ok(node) => {
                print_node(&node, i + 1);
            }
            Err(e) => {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    println!("Reached end of file after reading {} nodes", i);
                    break;
                } else {
                    return Err(e);
                }
            }
        }
    }

    Ok(())
}
