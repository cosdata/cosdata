use super::collection::Collection;
use super::tree_map::UnsafeVersionedItem;
use super::types::InternalId;
use super::versioning::VersionNumber;

/// Get the value from a versioned item chain at a specific version.
///
/// This function traverses the version chain of an `UnsafeVersionedItem` to find
/// the most recent value that was committed at or before the given version number.
///
/// # Arguments
/// * `versioned_item` - The head of the version chain to traverse
/// * `target_version` - The version number to query for
///
/// # Returns
/// * `Some(&T)` - The value that was current at the target version
/// * `None` - If no value exists at or before the target version
pub fn get_value_at_version<T>(
    versioned_item: &UnsafeVersionedItem<T>,
    target_version: VersionNumber,
) -> Option<&T> {
    // Start from the head of the chain (oldest version)
    let mut current = versioned_item;
    let mut result = None;

    // Traverse the chain to find the latest value at or before target_version
    loop {
        // If this version is at or before our target, it's a candidate
        if *current.version <= *target_version {
            result = Some(&current.value);
        } else {
            // This version is after our target, so we can't use it
            break;
        }

        // Move to the next version in the chain
        match unsafe { &*current.next.get() } {
            Some(next) => current = next,
            None => break,
        }
    }

    result
}

/// Fast O(1) vector count lookup using cached version-based counts
pub fn count_live_vectors(collection: &Collection, version: VersionNumber) -> usize {
    // Try cached count first - O(1)
    if let Some(count) = collection.get_cached_vector_count(version) {
        return count;
    }

    // Calculate incrementally and cache - O(versions_since_last_cache)
    let count = count_live_vectors_incremental(collection, version);
    collection.cache_vector_count(version, count);
    count
}

/// Incremental counting that builds on previous cached counts
fn count_live_vectors_incremental(collection: &Collection, target_version: VersionNumber) -> usize {
    // Find the most recent cached count before our target
    let (base_version, mut count) = collection.find_nearest_cached_count_before(target_version);

    // Get versions between base and target
    let versions_to_add = match collection.vcs.get_versions_starting_from_exclusive(base_version) {
        Ok(versions) => versions
            .into_iter()
            .filter(|v| v.version <= target_version)
            .map(|v| v.version)
            .collect::<Vec<_>>(),
        Err(_) => return count_live_vectors_by_range(collection, target_version),
    };

    // Add vectors from each intermediate version
    for version in versions_to_add {
        count += count_vectors_added_in_specific_version(collection, version);
    }

    count
}

/// Count vectors that were first added in a specific version (not cumulative)
fn count_vectors_added_in_specific_version(collection: &Collection, version: VersionNumber) -> usize {
    use std::sync::atomic::Ordering;

    let mut count = 0;
    let max_internal_id = collection.internal_id_counter.load(Ordering::Relaxed);

    for internal_id in 0..max_internal_id {
        let id = InternalId::from(internal_id);

        if let Some(versioned_item) = collection.internal_to_external_map.get_versioned(&id) {
            // Only count if this vector was first created in this exact version
            if versioned_item.version == version {
                count += 1;
            }
        }
    }

    count
}

/// Count live vectors using a brute-force approach by checking known internal IDs
///
/// This is the recommended implementation that works with the current TreeMap API.
/// It checks each internal ID up to the current counter value.
///
/// # Arguments
/// * `collection` - Reference to the collection to count vectors in
/// * `version` - The version number to query for live vectors
///
/// # Returns
/// * The number of unique live vectors at the specified version
pub fn count_live_vectors_by_range(collection: &Collection, version: VersionNumber) -> usize {
    use std::sync::atomic::Ordering;

    let mut count = 0;
    let max_internal_id = collection.internal_id_counter.load(Ordering::Relaxed);

    // Debug logging
    eprintln!(
        "DEBUG count_live_vectors: collection={}, version={}, max_internal_id={}",
        collection.meta.name, *version, max_internal_id
    );

    // Check each internal ID up to the maximum
    for internal_id in 0..max_internal_id {
        let id = InternalId::from(internal_id);

        // Get the versioned item for this internal ID
        if let Some(versioned_item) = collection.internal_to_external_map.get_versioned(&id) {
            // Check if this vector exists at the target version
            if get_value_at_version(versioned_item, version).is_some() {
                eprintln!(
                    "DEBUG: Found live vector at internal_id={}, version={}",
                    internal_id, *version
                );
                count += 1;
            }
        }
    }

    eprintln!("DEBUG count_live_vectors: Final count={}", count);
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::tree_map::UnsafeVersionedItem;

    #[test]
    fn test_get_value_at_version() {
        // Create a version chain: v1 -> v3 -> v5
        let item = UnsafeVersionedItem::new(VersionNumber::from(1), "value_v1");
        item.insert(VersionNumber::from(3), "value_v3");
        item.insert(VersionNumber::from(5), "value_v5");

        // Test queries at different versions
        assert_eq!(get_value_at_version(&item, VersionNumber::from(0)), None);
        assert_eq!(
            get_value_at_version(&item, VersionNumber::from(1)),
            Some(&"value_v1")
        );
        assert_eq!(
            get_value_at_version(&item, VersionNumber::from(2)),
            Some(&"value_v1")
        );
        assert_eq!(
            get_value_at_version(&item, VersionNumber::from(3)),
            Some(&"value_v3")
        );
        assert_eq!(
            get_value_at_version(&item, VersionNumber::from(4)),
            Some(&"value_v3")
        );
        assert_eq!(
            get_value_at_version(&item, VersionNumber::from(5)),
            Some(&"value_v5")
        );
        assert_eq!(
            get_value_at_version(&item, VersionNumber::from(10)),
            Some(&"value_v5")
        );
    }

    #[test]
    fn test_get_value_at_version_single_item() {
        let item = UnsafeVersionedItem::new(VersionNumber::from(5), "single_value");

        assert_eq!(get_value_at_version(&item, VersionNumber::from(4)), None);
        assert_eq!(
            get_value_at_version(&item, VersionNumber::from(5)),
            Some(&"single_value")
        );
        assert_eq!(
            get_value_at_version(&item, VersionNumber::from(10)),
            Some(&"single_value")
        );
    }
}
