//  Copyright 2024 Foyer Project Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

use std::{
    fmt::Debug,
    ops::{Deref, DerefMut, Range},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use bytes::{Buf, BufMut};
use foyer_common::strict_assert;
use parking_lot::RwLock;
use tokio::sync::{
    RwLock as AsyncRwLock, RwLockReadGuard as AsyncRwLockReadGuard, RwLockWriteGuard as AsyncRwLockWriteGuard,
};

use crate::{
    device::{Dev, IoBuffer, MonitoredDevice, RegionId},
    error::Result,
    serde::{Checksummer, KvInfo},
};

use super::bloom_filter::BloomFilterU64;

pub struct BucketAddress {
    region: RegionId,
    offset: u64,
    len: usize,
}

#[derive(Debug)]
pub struct Bucket {}

#[derive(Debug)]
struct BucketManagerInner {
    bloom_filters: Vec<Arc<RwLock<BloomFilterU64>>>,
    buckets: Vec<AsyncRwLock<Bucket>>,

    device: MonitoredDevice,
    bucket_size: usize,

    flush: bool,
}

#[derive(Debug, Clone)]
pub struct BucketManager {
    inner: Arc<BucketManagerInner>,
}

impl BucketManager {
    pub fn bloom_filter(&self, idx: usize) -> &Arc<RwLock<BloomFilterU64>> {
        &self.inner.bloom_filters[idx]
    }

    pub async fn read(&self, idx: usize) -> Result<BucketReadGuard<'_>> {
        let guard = self.inner.buckets[idx].read().await;
        let BucketAddress { region, offset, len } = self.locate(idx);
        let buffer = self.inner.device.read(region, offset, len).await?;
        let mut storage = BucketStorage::new(buffer, self.inner.bucket_size);

        if !storage.check() {
            storage.init();
        }

        Ok(BucketReadGuard { guard, storage })
    }

    pub async fn with<F>(&self, idx: usize, f: F) -> Result<()>
    where
        F: FnOnce(&mut BucketWriteGuard<'_>) + Send,
    {
        let mut guard = self.write(idx).await?;
        f(&mut guard);
        self.consume(guard).await?;
        Ok(())
    }

    async fn write(&self, idx: usize) -> Result<BucketWriteGuard<'_>> {
        let guard = self.inner.buckets[idx].write().await;
        let BucketAddress { region, offset, len } = self.locate(idx);
        let buffer = self.inner.device.read(region, offset, len).await?;
        let mut storage = BucketStorage::new(buffer, self.inner.bucket_size);

        if !storage.check() {
            storage.init();
        }

        Ok(BucketWriteGuard { guard, storage, idx })
    }

    async fn consume<'a>(&self, guard: BucketWriteGuard<'a>) -> Result<()> {
        let BucketWriteGuard {
            guard,
            mut storage,
            idx,
        } = guard;

        storage.update();
        let buffer = storage.into_inner();
        let BucketAddress { region, offset, len: _ } = self.locate(idx);
        self.inner.device.write(buffer, region, offset).await?;
        if self.inner.flush {
            self.inner.device.flush(Some(region)).await?;
        }

        drop(guard);

        Ok(())
    }

    pub fn buckets(&self) -> usize {
        self.inner.bloom_filters.len()
    }

    #[inline]
    fn locate(&self, idx: usize) -> BucketAddress {
        let region_buckets = self.inner.device.region_size() / self.inner.bucket_size;
        let region = idx / region_buckets;
        let offset = (idx % region_buckets) * self.inner.bucket_size;
        BucketAddress {
            region: region as _,
            offset: offset as _,
            len: self.inner.bucket_size,
        }
    }
}

pub struct BucketReadGuard<'a> {
    guard: AsyncRwLockReadGuard<'a, Bucket>,
    storage: BucketStorage,
}

impl<'a> Deref for BucketReadGuard<'a> {
    type Target = BucketStorage;

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

pub struct BucketWriteGuard<'a> {
    guard: AsyncRwLockWriteGuard<'a, Bucket>,
    storage: BucketStorage,
    idx: usize,
}

impl<'a> Deref for BucketWriteGuard<'a> {
    type Target = BucketStorage;

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

impl<'a> DerefMut for BucketWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.storage
    }
}

/// # Layout
///
/// Bucket:
///
///   metadata: 20B in total
///
/// ```plain
/// | checksum (4B) | timestamp (8B) |
/// | capacity (4B) | len (4B) |
/// | entry | entry | ... | entry |
/// ```
///
/// Explanations:
/// - checksum: checksum of all bytes after it.
/// - timestamp: last updated timestamp (us).
/// - capacity: capacity of the data section (metadata section excluded)
/// - len: length of the data secetion (metadata section excluded)
///
///
/// Entry:
///
/// ```plain
/// | len (4B) | data |s
/// ```
pub struct BucketStorage {
    /// bucket buffer
    buffer: IoBuffer,
    /// bucket size
    size: usize,
}

impl Debug for BucketStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BucketStorage").finish()
    }
}

impl BucketStorage {
    /// | checksum (4B) | timestamp (8B) |
    /// | capacity (4B) | len (4B) |
    const BUCKET_HEADER_LEN: usize = 20;
    /// | key len (4B) | value len (4B) |
    const ENTRY_HEADER_LEN: usize = 8;

    fn new(buffer: IoBuffer, size: usize) -> Self {
        Self { buffer, size }
    }

    fn into_inner(self) -> IoBuffer {
        self.buffer
    }

    fn init(&mut self) {
        self.set_capacity(self.size);
        self.set_len(0);
        self.update();
    }

    fn checksum(&self) -> u32 {
        (&self.buffer[0..4]).get_u32()
    }

    fn set_checksum(&mut self, checksum: u32) {
        (&mut self.buffer[0..4]).put_u32(checksum);
    }

    fn timestamp(&self) -> u64 {
        (&self.buffer[4..12]).get_u64()
    }

    fn set_timestamp(&mut self, timestamp: u64) {
        (&mut self.buffer[4..12]).put_u64(timestamp);
    }

    fn capacity(&self) -> usize {
        (&self.buffer[12..16]).get_u32() as _
    }

    fn set_capacity(&mut self, capacity: usize) {
        (&mut self.buffer[12..16]).put_u32(capacity as _)
    }

    fn len(&self) -> usize {
        (&self.buffer[16..20]).get_u32() as _
    }

    fn set_len(&mut self, len: usize) {
        (&mut self.buffer[16..20]).put_u32(len as _);
    }

    fn data(&self) -> &[u8] {
        &self.buffer[Self::BUCKET_HEADER_LEN..]
    }

    fn data_mut(&mut self) -> &mut [u8] {
        &mut self.buffer[Self::BUCKET_HEADER_LEN..]
    }

    /// bucket size
    fn size(&self) -> usize {
        self.size
    }

    pub fn insert(&mut self, info: KvInfo, data: &[u8]) {
        assert!(data.len() <= self.capacity());

        self.reserve(Self::ENTRY_HEADER_LEN + data.len());

        let mut cursor = self.len();
        (&mut self.data_mut()[cursor..cursor + 4]).put_u32(info.key_len as _);
        cursor += 4;
        (&mut self.data_mut()[cursor..cursor + 4]).put_u32(info.value_len as _);
        cursor += 4;
        (&mut self.data_mut()[cursor..cursor + data.len()]).put_slice(data);
        cursor += data.len();
        self.set_len(cursor);
        self.update();
    }

    fn reserve(&mut self, len: usize) {
        let mut remaining = self.capacity() - self.len();

        if len > remaining {
            let mut iter = BucketStorageIter::new(self);
            let start = loop {
                let slot = match iter.next() {
                    Some((_, entry)) => Self::ENTRY_HEADER_LEN + entry.len(),
                    None => break None,
                };
                remaining += slot;
                if remaining >= len {
                    break Some(iter.offset());
                }
            }
            .unwrap_or(self.len());
            let end = self.len();
            self.copy(start..end, 0);
            self.set_len(end - start);
        }
    }

    /// Copy and paste data in the data section.
    fn copy(&mut self, range: Range<usize>, dst: usize) {
        self.buffer.copy_within(
            Self::BUCKET_HEADER_LEN + range.start..Self::BUCKET_HEADER_LEN + range.end,
            Self::BUCKET_HEADER_LEN + dst,
        );
    }

    fn update(&mut self) {
        let checksum = Checksummer::checksum32(&self.buffer[4..self.buffer.len()]);
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64;

        self.set_timestamp(timestamp);
        self.set_checksum(checksum);
    }

    fn check(&self) -> bool {
        let expected = self.checksum();
        let get = Checksummer::checksum32(&self.buffer[4..self.buffer.len()]);
        expected == get
    }
}

#[derive(Debug)]
pub struct BucketStorageIter<'a> {
    storage: &'a BucketStorage,
    /// offset of the data section
    offset: usize,
}

impl<'a> BucketStorageIter<'a> {
    fn new(storage: &'a BucketStorage) -> Self {
        Self { storage, offset: 0 }
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn is_valid(&self) -> bool {
        // A least a entry header len is remaining.
        self.offset + BucketStorage::ENTRY_HEADER_LEN < std::cmp::min(self.storage.len(), self.storage.size())
    }

    /// current kv info
    ///
    /// # Panics
    ///
    /// Panics if the iter is on a invalid position.
    fn info(&self) -> KvInfo {
        strict_assert!(self.is_valid());
        let key_len = (&self.storage.data()[self.offset..self.offset + 4]).get_u32() as _;
        let value_len = (&self.storage.data()[self.offset + 4..self.offset + 8]).get_u32() as _;
        KvInfo { key_len, value_len }
    }

    /// current slot size (entry header + entry data)
    ///
    /// # Panics
    ///
    /// Panics if the iter is on a invalid position.
    fn slot(&self) -> usize {
        let info = self.info();
        BucketStorage::ENTRY_HEADER_LEN + info.key_len + info.value_len
    }

    fn entry(&self) -> Option<(KvInfo, &'a [u8])> {
        if !self.is_valid() {
            return None;
        }

        let info = self.info();
        let start = self.offset + BucketStorage::ENTRY_HEADER_LEN;
        let end = start + info.key_len + info.value_len;

        if end > self.storage.len() {
            return None;
        }
        Some((info, &self.storage.data()[start..end]))
    }

    fn step(&mut self) {
        if !self.is_valid() {
            return;
        }
        self.offset += self.slot()
    }
}

impl<'a> Iterator for BucketStorageIter<'a> {
    type Item = (KvInfo, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if !self.is_valid() {
            return None;
        }

        let entry = self.entry();
        self.step();

        entry
    }
}
