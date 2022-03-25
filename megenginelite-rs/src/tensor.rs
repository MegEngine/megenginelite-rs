//! The tensor module

use super::*;
use crate::ffi::*;

#[doc(hidden)]
#[derive(Debug)]
pub struct SliceInfo<'a> {
    pub start: &'a [usize],
    pub end: &'a [Option<usize>],
    pub step: &'a [usize],
}

/// The simple layout description
///
/// see also [`crate::DataType`], which is the alias of `LiteDataType`
#[derive(Debug)]
pub struct Layout<'a> {
    pub shapes: &'a [usize],
    pub data_type: LiteDataType,
}

impl<'a> Default for Layout<'a> {
    fn default() -> Self {
        Layout {
            shapes: &[],
            data_type: DataType::F32,
        }
    }
}

/// The Lite Tensor object
pub struct Tensor {
    inner: LiteTensor,
    desc: LiteTensorDesc,
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl<'a> Layout<'a> {
    pub(crate) fn as_raw(&self) -> LiteLayout {
        let mut shapes = [0; LAYOUT_MAX_DIM as usize];
        for (i, v) in self.shapes.iter().enumerate() {
            shapes[i] = *v;
        }
        LiteLayout {
            data_type: self.data_type,
            ndim: self.shapes.len(),
            shapes,
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            api().LITE_destroy_tensor(self.inner);
        }
    }
}

impl Tensor {
    pub(crate) fn new(inner: LiteTensor, desc: LiteTensorDesc) -> Tensor {
        Tensor { inner, desc }
    }

    /// The storage memory of tensor is host memory.
    pub fn host() -> LiteResult<Tensor> {
        let mut inner = std::ptr::null_mut();
        let desc;
        unsafe {
            desc = LiteTensorDesc {
                is_pinned_host: 0,
                layout: Self::default_layout(),
                device_type: DeviceType::CPU,
                device_id: 0,
            };
            api().LITE_make_tensor(desc, &mut inner).into_rst()?;
        };
        Ok(Tensor { inner, desc })
    }

    /// The storage memory of the tensor is pinned memory, this is used
    /// to optimize the H2D or D2H memory copy
    ///
    /// see also [`crate::DeviceType`], which is the alias of `LiteDeviceType`
    pub fn pinned_host(ty: LiteDeviceType, dev_id: i32) -> LiteResult<Tensor> {
        let mut inner = std::ptr::null_mut();
        let desc;
        unsafe {
            desc = LiteTensorDesc {
                is_pinned_host: 1,
                layout: Self::default_layout(),
                device_type: ty,
                device_id: dev_id,
            };
            api().LITE_make_tensor(desc, &mut inner).into_rst()?;
        };
        Ok(Tensor { inner, desc })
    }

    /// The storage memory of tensor is device memory.
    ///
    /// see also [`crate::DeviceType`], which is the alias of `LiteDeviceType`
    pub fn device(ty: LiteDeviceType, dev_id: i32) -> LiteResult<Tensor> {
        let mut inner = std::ptr::null_mut();
        let desc;
        unsafe {
            desc = LiteTensorDesc {
                is_pinned_host: 0,
                layout: Self::default_layout(),
                device_type: ty,
                device_id: dev_id,
            };
            api().LITE_make_tensor(desc, &mut inner).into_rst()?;
        };
        Ok(Tensor { inner, desc })
    }

    pub fn shape(&self) -> &[usize] {
        &self.desc.layout.shapes[..self.desc.layout.ndim as usize]
    }

    /// see also [`crate::DataType`], which is the alias of `LiteDataType`
    pub fn dtype(&self) -> LiteDataType {
        self.desc.layout.data_type
    }

    /// Change the layout of a Tensor object.
    pub fn set_layout(&mut self, layout: Layout) {
        let layout = layout.as_raw();
        self.desc.layout = layout;
        unsafe { api().LITE_set_tensor_layout(self.inner, layout) };
    }

    /// Get the tensor capacity in byte of a Tensor object.
    pub fn nbytes(&self) -> usize {
        let mut length = 0;
        unsafe { api().LITE_get_tensor_total_size_in_byte(self.inner, &mut length) };
        length
    }

    /// Whether the tensor memory is continue.
    pub fn is_continue(&self) -> bool {
        let mut is_continue = 0i32;
        unsafe { api().LITE_is_memory_continue(self.inner, &mut is_continue) };
        is_continue != 0
    }

    pub fn dev_id(&self) -> i32 {
        self.desc.device_id
    }

    /// see also [`crate::DeviceType`], which is the alias of `LiteDeviceType`
    pub fn dev_type(&self) -> LiteDeviceType {
        self.desc.device_type
    }

    pub fn is_pinned_host(&self) -> bool {
        self.desc.is_pinned_host != 0
    }

    pub fn is_host(&self) -> bool {
        self.is_pinned_host() || self.dev_type() == DeviceType::CPU
    }

    /// Reshape a tensor with the memroy not change, the total number of
    /// element in the reshaped tensor must equal to the origin tensor, the input
    /// shape must only contain one or zero -1 to flag it can be deduced automatically.
    pub fn reshape(&mut self, shape: &[i32]) {
        unsafe {
            api().LITE_tensor_reshape(self.inner, shape.as_ptr(), shape.len() as i32);
            api().LITE_get_tensor_layout(self.inner, &mut self.desc.layout);
        };
    }

    /// Fill zero to the tensor
    pub fn fill_zero(&mut self) {
        unsafe { api().LITE_tensor_fill_zero(self.inner) };
    }

    /// Slice a tensor with input param, see also [`crate::idx!()`]
    ///
    /// # Example
    /// ```no_run
    /// # use megenginelite_rs::*;
    /// let mut t = Tensor::host().unwrap();
    /// t.set_layout(Layout {
    ///     data_type: DataType::U8,
    ///     shapes: &[1, 5],
    /// });
    /// t.slice(idx![0, 0..2;2]);
    /// ```
    pub fn slice(&self, info: SliceInfo) -> Tensor {
        let mut desc = self.desc;
        let mut inner = std::ptr::null_mut();
        let end: Vec<_> = info
            .end
            .iter()
            .enumerate()
            .map(|(i, x)| x.unwrap_or(self.desc.layout.shapes[i]))
            .collect();
        unsafe {
            api().LITE_tensor_slice(
                self.inner,
                info.start.as_ptr(),
                end.as_ptr(),
                info.step.as_ptr(),
                info.start.len(),
                &mut inner,
            );
            api().LITE_get_tensor_layout(inner, &mut desc.layout);
        };
        Tensor { inner, desc }
    }

    /// Copy tensor form other tensor
    pub fn copy_from(&mut self, other: &Tensor) {
        unsafe {
            api().LITE_tensor_copy(self.inner, other.inner);
            api().LITE_get_tensor_layout(self.inner, &mut self.desc.layout);
        };
    }

    /// Get the memory pointer of a Tensor object.
    pub fn as_ptr<T>(&self) -> *const T {
        let mut p = std::ptr::null_mut();
        unsafe {
            api().LITE_get_tensor_memory(self.inner, &mut p);
        }
        p as *const T
    }

    /// Get the memory mutable pointer of a Tensor object.
    pub fn as_ptr_mut<T>(&mut self) -> *mut T {
        let mut p = std::ptr::null_mut();
        unsafe {
            api().LITE_get_tensor_memory(self.inner, &mut p);
        }
        p as *mut T
    }

    /// As a slice
    ///
    /// # Panic
    /// if the tensor is not a host tensor
    pub fn as_slice<T>(&self) -> &[T] {
        if !self.is_host() {
            panic!("as_slice only support for host tensor")
        }
        unsafe {
            std::slice::from_raw_parts(self.as_ptr(), self.nbytes() / std::mem::size_of::<T>())
        }
    }

    /// As a mutable slice
    ///
    /// # Panic
    /// if the tensor is not a host tensor
    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        if !self.is_host() {
            panic!("as_slice_mut only support for host tensor")
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.as_ptr_mut(),
                self.nbytes() / std::mem::size_of::<T>(),
            )
        }
    }

    /// As a [`ndarray::ArrayView`]
    ///
    /// # Panic
    /// if the tensor is not a host tensor
    #[cfg(feature = "ndarray-basis")]
    pub fn as_ndarray<T>(&self) -> ndarray::ArrayView<T, ndarray::IxDyn> {
        ndarray::ArrayView::from_shape(self.shape(), self.as_slice()).unwrap()
    }

    /// As a [`ndarray::ArrayViewMut`]
    ///
    /// # Panic
    /// if the tensor is not a host tensor
    #[cfg(feature = "ndarray-basis")]
    pub fn as_ndarray_mut<T>(&mut self) -> ndarray::ArrayViewMut<T, ndarray::IxDyn> {
        let shape = self.shape();
        let p: *const T = self.as_ptr();
        unsafe { ndarray::ArrayViewMut::from_shape_ptr(shape, p as *mut _) }
    }

    /// Borrow the memory from the `other`, the self memory will be freed
    pub fn borrow_from<'a, 'b: 'a>(&'b mut self, other: &'a Tensor) {
        unsafe {
            api().LITE_tensor_share_memory_with(self.inner, other.inner);
            api().LITE_get_tensor_layout(self.inner, &mut self.desc.layout);
        }
    }

    /// Use the user allocated data to reset the memory of the tensor
    /// `p` The allocated memory which satisfy the Tensor
    /// `length` The length of the allocated memory
    ///
    /// # Safety
    /// the memory will not be managed by the lite, later, the user should delete it.
    pub unsafe fn borrow_from_raw_parts<T>(&mut self, p: *mut T, length: usize) {
        let nbytes = length * std::mem::size_of::<T>();
        assert_eq!(nbytes, self.nbytes());
        api().LITE_reset_tensor_memory(self.inner, p as *mut std::ffi::c_void, nbytes);
    }

    pub(crate) fn default_layout() -> LiteLayout {
        LiteLayout {
            ndim: 0,
            data_type: super::DataType::U8,
            shapes: [0; 7],
        }
    }
}

#[cfg(test)]
mod test {
    use crate::*;

    fn get_tensor(h: usize, w: usize) -> Tensor {
        let mut tensor = Tensor::host().unwrap();
        tensor.set_layout(Layout {
            data_type: DataType::U8,
            shapes: &[h, w],
        });
        tensor
    }

    #[test]
    fn test_basis() {
        let tensor = get_tensor(100, 200);
        assert!(!tensor.is_pinned_host());
        assert_eq!(tensor.dev_type(), DeviceType::CPU);
        assert_eq!(tensor.nbytes(), 20000);
        assert_eq!(tensor.shape()[0], 100);
        assert_eq!(tensor.shape()[1], 200);
        assert!(tensor.is_continue());
    }

    #[test]
    fn test_slice() {
        let tensor = get_tensor(100, 200);
        let sub = tensor.slice(idx![0, 0..100]);
        assert_eq!(sub.shape()[0], 1);
        assert_eq!(sub.shape()[1], 100);
        assert!(sub.is_continue());

        let sub = tensor.slice(idx![10..50, 50..100]);
        assert_eq!(sub.shape()[0], 40);
        assert_eq!(sub.shape()[1], 50);
        assert!(!sub.is_continue());
    }

    #[test]
    fn test_fill_zero() {
        let mut tensor = get_tensor(10, 20);
        tensor.fill_zero();
        for &i in tensor.as_slice::<u8>() {
            assert_eq!(i, 0);
        }
    }

    #[test]
    fn test_copy_from() {
        let mut tensor = get_tensor(10, 20);
        let slice = tensor.as_slice_mut::<u8>();
        slice.iter_mut().enumerate().for_each(|(i, x)| {
            *x = i as u8;
        });
        let mut other = Tensor::host().unwrap();
        other.copy_from(&tensor);
        let zip = tensor
            .as_slice::<u8>()
            .iter()
            .zip(other.as_slice::<u8>().iter());
        for (a, b) in zip {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_borrow_from() {
        let other_tensor = get_tensor(10, 20);
        let mut tensor = get_tensor(10, 20);

        tensor.borrow_from(&other_tensor);

        assert_eq!(other_tensor.as_ptr::<u8>(), tensor.as_ptr::<u8>());
    }
}
