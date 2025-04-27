use std::convert::TryInto;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Opt {
    input: PathBuf,
    level0_neighbors: u16,
    leveln_neighbors: u16,
    skip_bytes: u64,
    limit: usize,
}

impl Opt {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 5 {
            eprintln!("Usage: binary-parser <input> <level0_neighbors> <leveln_neighbors> <skip_bytes> <limit>");
            std::process::exit(1);
        }

        Self {
            input: PathBuf::from(&args[1]),
            level0_neighbors: args[2].parse().unwrap_or(0),
            leveln_neighbors: args[3].parse().unwrap_or(0),
            skip_bytes: args[4].parse().unwrap_or(0),
            limit: args.get(5).map(|s| s.parse().unwrap_or(1)).unwrap_or(1),
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
struct VersionPtr {
    offset: u32,
    version_number: u16,
    version_hash: u32,
}

#[derive(Debug)]
struct LinkValidation {
    is_valid: bool,
    reason: Option<String>,
}

impl VersionPtr {
    fn read(reader: &mut impl Read) -> io::Result<Self> {
        let mut buffer = [0u8; 10];
        reader.read_exact(&mut buffer)?;

        Ok(VersionPtr {
            offset: u32::from_le_bytes(buffer[0..4].try_into().unwrap()),
            version_number: u16::from_le_bytes(buffer[4..6].try_into().unwrap()),
            version_hash: u32::from_le_bytes(buffer[6..10].try_into().unwrap()),
        })
    }

    fn is_absent(&self) -> bool {
        self.offset == u32::MAX
    }

    fn validate_link(
        &self,
        current_file: &Path,
        current_hash: u32,
        current_level: u8,
        target_level: u8,
    ) -> LinkValidation {
        if self.is_absent() {
            return LinkValidation {
                is_valid: true,
                reason: None,
            };
        }

        // Get current file's directory and version
        let dir = current_file.parent().unwrap_or_else(|| Path::new("."));

        // Determine neighbor count based on target level, not current level
        let neighbor_count = if target_level == 0 {
            unsafe { LEVEL0_NEIGHBORS }
        } else {
            unsafe { LEVELN_NEIGHBORS }
        };
        let node_size = Self::get_serialized_size(neighbor_count); // Updated size calculation

        // Check if offset is aligned with the target level's node size
        if self.offset as u64 % node_size != 0 {
            return LinkValidation {
                is_valid: false,
                reason: Some(format!(
                    "Offset {} is not aligned to level {} node size {}",
                    self.offset, target_level, node_size
                )),
            };
        }

        // If same hash, just check current file
        if self.version_hash == current_hash && current_level == target_level {
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
        let version_file = if target_level == 0 {
            dir.join(format!("{}_0.index", self.version_hash))
        } else {
            dir.join(format!("{}.index", self.version_hash))
        };

        if !version_file.exists() {
            return LinkValidation {
                is_valid: false,
                reason: Some(format!("Version file {:?} not found", version_file)),
            };
        }

        // Check file size
        let file_size = std::fs::metadata(&version_file)
            .map(|m| m.len())
            .unwrap_or(0);

        if (self.offset as u64) >= file_size {
            return LinkValidation {
                is_valid: false,
                reason: Some(format!(
                    "Offset {} exceeds file size {} in {:?}",
                    self.offset, file_size, version_file
                )),
            };
        }

        LinkValidation {
            is_valid: true,
            reason: None,
        }
    }

    // New method to calculate serialized size based on neighbor count
    fn get_serialized_size(neighbor_count: u16) -> u64 {
        // Formula from node.rs: nb * 19 + 49 (where nb is neighbor count)
        (neighbor_count as u64) * 19 + 47 + 2 // 47 bytes for properties + 2 bytes for neighbor length
    }
}

static mut LEVEL0_NEIGHBORS: u16 = 0;
static mut LEVELN_NEIGHBORS: u16 = 0;

#[derive(Debug)]
struct Neighbor {
    id: u32,
    ptr: VersionPtr,
    distance: Distance,
    validation: LinkValidation,
}

#[derive(Debug)]
struct Node {
    level: u8,
    prop_offset: u32,
    prop_length: u32,
    metadata_offset: u32,
    metadata_length: u32,
    parent: VersionPtr,
    parent_validation: LinkValidation,
    child: VersionPtr,
    child_validation: LinkValidation,
    root_version: VersionPtr,
    root_version_validation: LinkValidation,
    root_version_is_root: bool,
    neighbors: Vec<Option<Neighbor>>,
}

fn read_neighbors(
    reader: &mut impl Read,
    current_file: &Path,
    current_hash: u32,
    expected_length: u16,
    node_level: u8,
) -> io::Result<Vec<Option<Neighbor>>> {
    let mut actual_length = [0u8; 2];
    reader.read_exact(&mut actual_length)?;
    let actual_length = u16::from_le_bytes(actual_length);

    if actual_length != expected_length {
        println!(
            "Warning: Expected neighbors length {}, but found {}",
            expected_length, actual_length
        );
    }

    let mut neighbors = Vec::with_capacity(expected_length as usize);
    for _ in 0..expected_length {
        let mut id_buffer = [0u8; 4];
        reader.read_exact(&mut id_buffer)?;
        let id = u32::from_le_bytes(id_buffer);

        if id == u32::MAX {
            let mut skip_buffer = [0u8; 15];
            reader.read_exact(&mut skip_buffer)?;
            neighbors.push(None);
            continue;
        }

        let ptr = VersionPtr::read(reader)?;
        let validation = ptr.validate_link(current_file, current_hash, node_level, node_level);

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

fn read_node(file: &mut File, current_file: &Path, current_hash: u32) -> io::Result<Node> {
    let mut level = [0u8; 1];
    file.read_exact(&mut level)?;

    let neighbor_count = unsafe {
        if level[0] == 0 {
            LEVEL0_NEIGHBORS
        } else {
            LEVELN_NEIGHBORS
        }
    };

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

    let parent = VersionPtr::read(file)?;
    let parent_validation =
        parent.validate_link(current_file, current_hash, level[0], level[0] + 1);

    let child = VersionPtr::read(file)?;
    let child_validation =
        child.validate_link(current_file, current_hash, level[0], level[0].max(1) - 1);

    let root_version = VersionPtr::read(file)?;

    // Check if root version is tagged as self
    let root_version_is_root = !root_version.is_absent()
        && (root_version.offset % VersionPtr::get_serialized_size(neighbor_count) as u32 != 0);

    // Adjust offset if it's tagged
    let mut adjusted_root_version = root_version;
    if root_version_is_root {
        adjusted_root_version.offset -= 1;
    }

    let root_version_validation =
        adjusted_root_version.validate_link(current_file, current_hash, level[0], level[0]);

    let neighbors = read_neighbors(file, current_file, current_hash, neighbor_count, level[0])?;

    Ok(Node {
        level: level[0],
        prop_offset,
        prop_length,
        metadata_offset,
        metadata_length,
        parent,
        parent_validation,
        child,
        child_validation,
        root_version: adjusted_root_version,
        root_version_validation,
        root_version_is_root,
        neighbors,
    })
}

fn print_version_ptr(ptr: &VersionPtr, validation: &LinkValidation, prefix: &str) {
    if ptr.is_absent() {
        println!("{}: absent", prefix);
        return;
    }

    print!(
        "{}: offset={}, version={}, hash={}",
        prefix, ptr.offset, ptr.version_number, ptr.version_hash
    );

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

    print_version_ptr(&node.parent, &node.parent_validation, "Parent");
    print_version_ptr(&node.child, &node.child_validation, "Child");

    // Display the root version with its tag info
    if node.root_version.is_absent() {
        println!("Root version: absent");
    } else {
        print!(
            "Root version: offset={}, version={}, hash={}, is_root={}",
            node.root_version.offset,
            node.root_version.version_number,
            node.root_version.version_hash,
            node.root_version_is_root
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
                    "  {}. ID: {}, Version: offset={}, version={}, hash={}, Distance: {:?}",
                    i + 1,
                    n.id,
                    n.ptr.offset,
                    n.ptr.version_number,
                    n.ptr.version_hash,
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

    // Set global neighbor counts
    unsafe {
        LEVEL0_NEIGHBORS = opt.level0_neighbors;
        LEVELN_NEIGHBORS = opt.leveln_neighbors;
    }

    let mut file = File::open(&opt.input)?;

    // Extract version hash from filename
    let current_hash = opt
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.split('_').next())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    if opt.skip_bytes > 0 {
        file.seek(SeekFrom::Start(opt.skip_bytes))?;
    }

    for i in 0..opt.limit {
        match read_node(&mut file, &opt.input, current_hash) {
            Ok(node) => {
                let node_size = VersionPtr::get_serialized_size(if node.level == 0 {
                    opt.level0_neighbors
                } else {
                    opt.leveln_neighbors
                });

                print_node(&node, i + 1);

                if i < opt.limit - 1 {
                    let current_pos = file.stream_position()?;
                    let next_aligned_pos = ((current_pos + node_size - 1) / node_size) * node_size;
                    file.seek(SeekFrom::Start(next_aligned_pos))?;
                }
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
