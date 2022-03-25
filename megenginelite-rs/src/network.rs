//! The network module

use super::{api, IntoLiteRst, LiteResult, NetworkBuilder, Tensor};
use crate::ffi::*;
use atomic_waker::AtomicWaker;
use std::ffi::{CStr, CString};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

/// The network is construct from a model, implement model load, init, forward, and display some
/// model information
pub struct Network {
    pub(super) inner: LiteNetwork,
    waker: Arc<State>,
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe {
            api().LITE_destroy_network(self.inner);
        }
    }
}

unsafe impl Send for Network {}
unsafe impl Sync for Network {}

impl Network {
    pub(super) fn new(inner: LiteNetwork) -> Network {
        Network {
            inner,
            waker: Arc::new(State::new()),
        }
    }

    /// Get a builder to build network
    pub fn builder<'a>() -> NetworkBuilder<'a> {
        NetworkBuilder::default()
    }

    /// Forward the network with filled input data and fill the output data
    /// , and wait until forward finish in sync model
    pub fn exec_wait(&mut self) -> LiteResult<()> {
        unsafe {
            api().LITE_forward(self.inner).into_rst()?;
            api().LITE_wait(self.inner).into_rst()?;
        }
        Ok(())
    }

    /// Async version of `exec_wait`
    pub fn exec(&mut self) -> AsyncExec {
        self.waker.reset();
        unsafe extern "C" fn callback(user_data: *mut std::ffi::c_void) -> i32 {
            let waker = user_data as *mut State;
            let waker = waker.as_ref().unwrap();
            waker.signal(0); // todo(wangyi): check error
            0
        }
        let code = unsafe {
            api().LITE_set_async_callback_with_userdata(
                self.inner,
                Some(callback),
                Arc::as_ptr(&self.waker) as *mut std::ffi::c_void,
            );

            api().LITE_forward(self.inner)
        };
        self.waker.rlt.store(code, Ordering::Relaxed);
        AsyncExec {
            state: self.waker.clone(),
        }
    }

    /// Get the network input and ouput tensor
    pub fn io_tensor(&self, name: &str) -> Option<Tensor> {
        let name = CString::new(name).unwrap();
        let mut tensor = std::ptr::null_mut();
        let mut desc;

        unsafe {
            desc = LiteTensorDesc {
                is_pinned_host: 0,
                layout: Tensor::default_layout(),
                device_type: LiteDeviceType_LITE_CPU,
                device_id: 0,
            };
            api().LITE_get_io_tensor(
                self.inner,
                name.as_ptr(),
                LiteTensorPhase_LITE_IO,
                &mut tensor,
            );
            api().LITE_is_pinned_host(tensor, &mut desc.is_pinned_host);
            api().LITE_get_tensor_device_type(tensor, &mut desc.device_type);
            api().LITE_get_tensor_layout(tensor, &mut desc.layout);
            api().LITE_get_tensor_device_id(tensor, &mut desc.device_id);
        }

        if tensor.is_null() {
            None
        } else {
            Some(Tensor::new(tensor, desc))
        }
    }

    /// Get the input tensor name in the order in loaded model
    pub fn input_names(&self) -> Vec<&str> {
        let mut n = 0;
        let mut names;
        unsafe {
            api().LITE_get_all_input_name(self.inner, &mut n, std::ptr::null_mut());
            names = vec![std::ptr::null(); n as usize];
            if n > 0 {
                api().LITE_get_all_input_name(self.inner, &mut n, names.as_mut_ptr());
            }
        };
        names
            .iter()
            .map(|x| unsafe { CStr::from_ptr(*x) }.to_str().unwrap())
            .collect()
    }

    /// Get the output tensor name in the order in loaded model
    pub fn output_names(&self) -> Vec<&str> {
        let mut n = 0;
        let mut names;
        unsafe {
            api().LITE_get_all_output_name(
                self.inner,
                std::ptr::addr_of_mut!(n),
                std::ptr::null_mut(),
            );
            names = vec![std::ptr::null(); n as usize];
            if n > 0 {
                api().LITE_get_all_output_name(
                    self.inner,
                    std::ptr::addr_of_mut!(n),
                    names.as_mut_ptr(),
                );
            }
        };
        names
            .iter()
            .map(|x| unsafe { CStr::from_ptr(*x) }.to_str().unwrap())
            .collect()
    }
}

#[doc(hidden)]
#[derive(Default, Clone)]
pub struct AsyncExec {
    state: Arc<State>,
}

#[derive(Default)]
struct State {
    rlt: AtomicI32,
    finish: AtomicBool,
    waker: AtomicWaker,
}

impl State {
    fn new() -> Self {
        State {
            waker: AtomicWaker::new(),
            rlt: AtomicI32::new(0),
            finish: AtomicBool::new(false),
        }
    }

    fn signal(&self, rlt: i32) {
        self.rlt.store(rlt, Ordering::Relaxed);
        self.finish.store(true, Ordering::Relaxed);
        self.waker.wake();
    }

    fn reset(&self) {
        self.rlt.store(0, Ordering::Relaxed);
        self.finish.store(false, Ordering::Relaxed);
    }
}

impl Future for AsyncExec {
    type Output = LiteResult<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let view = self.state.rlt.load(Ordering::Relaxed);
        if view != 0 {
            Poll::Ready(view.into_rst())
        } else {
            self.state.waker.register(cx.waker());
            if self.state.finish.load(Ordering::Relaxed) {
                Poll::Ready(self.state.rlt.load(Ordering::Relaxed).into_rst())
            } else {
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn test_basis() -> LiteResult<()> {
        let mut network = Network::builder().build(model_path())?;
        assert!(network.io_tensor("data").is_some());
        if let Some(input) = network.io_tensor("data") {
            assert_eq!(input.dtype(), DataType::F32);
            assert_eq!(input.shape(), &[1, 3, 224, 224]);
        }
        network.exec_wait()?;
        Ok(())
    }

    #[tokio::test]
    async fn test_async() -> LiteResult<()> {
        let mut network = Network::builder().build(model_path())?;
        assert!(network.io_tensor("data").is_some());
        if let Some(input) = network.io_tensor("data") {
            assert_eq!(input.dtype(), DataType::F32);
            assert_eq!(input.shape(), &[1, 3, 224, 224]);
        }
        network.exec().await?;
        Ok(())
    }

    #[test]
    fn test_io() -> LiteResult<()> {
        let mut network = Network::builder().build(model_path())?;
        assert!(network.io_tensor("data").is_some());
        if let Some(input) = network.io_tensor("data") {
            assert_eq!(input.dtype(), DataType::F32);
            assert_eq!(input.shape(), &[1, 3, 224, 224]);
        }
        network.exec_wait()?;
        Ok(())
    }
}
