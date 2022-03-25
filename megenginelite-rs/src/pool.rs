use super::ffi::LiteDeviceType;
use super::tensor::*;
use super::{idx, LiteResult};

/// A tensor pool to reuse memory
///
/// # Example
/// ```
/// # use megenginelite_rs::*;
/// # #[tokio::main]
/// # async fn main() -> LiteResult<()> {
/// let pool = TensorPool::host(Layout {
///    data_type: DataType::U8,
///    shapes: &[10, 10, 10],
/// })?;
/// let free_n = pool.free_n();
/// {
///     let idx = pool.get().await;
///     let tensor = pool.at(&idx);
///     assert_eq!(free_n, pool.free_n() + 1);
/// }
/// assert_eq!(free_n, pool.free_n());
/// # Ok(())
/// # }
/// ```
pub struct TensorPool {
    mem: Tensor,
    phead: *mut u8,
    freelist: FreeList,
}

unsafe impl std::marker::Send for TensorPool {}
unsafe impl Sync for TensorPool {}

impl TensorPool {
    fn new(mut mem: Tensor, freelist: FreeList) -> Self {
        Self {
            phead: mem.as_ptr_mut(),
            mem,
            freelist,
        }
    }
    /// Return the number of free blocks in pool
    #[inline]
    pub fn free_n(&self) -> usize {
        self.freelist.len()
    }
    /// Create a pool with host memory, see also [`Tensor::host()`]
    pub fn host(layout: Layout) -> LiteResult<Self> {
        let freelist = FreeList::new(layout.shapes[0] as usize);
        let mut mem = Tensor::host()?;
        mem.set_layout(layout);
        Ok(Self::new(mem, freelist))
    }
    /// Create a pool with device memory, see also [`Tensor::device()`]
    pub fn device(ty: LiteDeviceType, dev_id: i32, layout: Layout) -> LiteResult<Self> {
        let freelist = FreeList::new(layout.shapes[0] as usize);
        let mut mem = Tensor::device(ty, dev_id)?;
        mem.set_layout(layout);
        Ok(Self::new(mem, freelist))
    }
    /// Create a pool with pinned host memory, see also [`Tensor::pinned_host()`]
    pub fn pinned_host(ty: LiteDeviceType, dev_id: i32, layout: Layout) -> LiteResult<Self> {
        let freelist = FreeList::new(layout.shapes[0] as usize);
        let mut mem = Tensor::pinned_host(ty, dev_id)?;
        mem.set_layout(layout);
        Ok(Self::new(mem, freelist))
    }
    /// Get the data pointer of the inner tensor
    pub fn as_ptr<T>(&self) -> *const T {
        self.phead as _
    }
    /// Get inner tensor
    pub fn as_tensor(&self) -> &Tensor {
        &self.mem
    }
    /// Request an index, will block if the pool is empty
    pub async fn get(&self) -> Idx {
        self.freelist.pop().await
    }
    /// Get the tensor at `idx`
    pub fn at(&self, idx: &Idx) -> Tensor {
        self.mem.slice(idx![idx.get()])
    }
}

use async_channel::*;
use std::{fmt::Display, ops::Deref};

/// An tensor index of the [`TensorPool`]
pub struct Idx {
    id: usize,
    s: Sender<usize>,
}

impl Idx {
    /// Get index.
    #[inline]
    pub fn get(&self) -> usize {
        self.id
    }
}

impl Display for Idx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.id.fmt(f)
    }
}

impl Deref for Idx {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

impl Drop for Idx {
    fn drop(&mut self) {
        // never block because the number of datas is equal to the capacity of queue
        self.s.try_send(self.id).ok();
    }
}

struct FreeList {
    s: Sender<usize>,
    r: Receiver<usize>,
}

impl FreeList {
    fn new(n: usize) -> FreeList {
        let (s, r) = bounded(n);
        for i in 0..n {
            s.try_send(i).ok();
        }
        assert_eq!(s.len(), n);
        FreeList { s, r }
    }

    #[inline]
    async fn pop(&self) -> Idx {
        Idx {
            id: self.r.recv().await.unwrap(),
            s: self.s.clone(),
        }
    }
    #[inline]
    fn len(&self) -> usize {
        self.s.len()
    }
}
