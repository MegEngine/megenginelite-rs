use super::{api, NetworkBuilder, Tensor};
use atomic_waker::AtomicWaker;
use megenginelite_sys::*;
use std::ffi::{CStr, CString};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

pub struct Network {
    pub(super) inner: LiteNetwork,
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe {
            api().LITE_destroy_network(self.inner);
        }
    }
}

impl Network {
    /// Get a builder to build network
    pub fn builder<'a>() -> NetworkBuilder<'a> {
        NetworkBuilder {
            config: None,
            io: None,
            option_setting: vec![],
            phantom: Default::default(),
        }
    }

    /// Forward the network with filled input data and fill the output data
    /// , and wait until forward finish in sync model
    pub fn exec_wait(&mut self) {
        unsafe {
            api().LITE_forward(self.inner);
            api().LITE_wait(self.inner);
        }
    }

    /// Async version of `exec_wait`
    pub async fn exec(&mut self) {
        unimplemented!() // megenginelite dont support callback with userdata
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
        let mut n = 0u64;
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
        let mut n = 0u64;
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

#[derive(Default, Clone)]
pub struct AsyncExec {
    state: Arc<State>,
}

#[derive(Default)]
struct State {
    finish: AtomicBool,
    waker: AtomicWaker,
}

impl AsyncExec {
    pub fn new() -> Self {
        AsyncExec {
            state: Arc::new(State {
                waker: AtomicWaker::new(),
                finish: AtomicBool::new(false),
            }),
        }
    }

    pub fn signal(&self) {
        self.state.finish.store(true, Ordering::Relaxed);
        self.state.waker.wake();
    }
}

impl Future for AsyncExec {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.state.finish.load(Ordering::Relaxed) {
            Poll::Ready(())
        } else {
            self.state.waker.register(cx.waker());
            if self.state.finish.load(Ordering::Relaxed) {
                Poll::Ready(())
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
    fn test_basis() {
        unsafe { assert!(load(lib_path()).is_ok()) };
        let mut network = Network::builder().build(model_path());
        assert!(network.io_tensor("data").is_some());
        if let Some(input) = network.io_tensor("data") {
            assert_eq!(input.dtype(), DataType::F32);
            assert_eq!(input.shape(), &[1, 3, 224, 224]);
        }
        network.exec_wait();
    }

    #[test]
    fn test_io() {
        unsafe { assert!(load(lib_path()).is_ok()) };
        let io = NetworkIO {
            inputs: &[],
            outputs: &[],
        };
        let mut network = Network::builder().io(io).build(model_path());
        assert!(network.io_tensor("data").is_some());
        if let Some(input) = network.io_tensor("data") {
            assert_eq!(input.dtype(), DataType::F32);
            assert_eq!(input.shape(), &[1, 3, 224, 224]);
        }
        network.exec_wait();
    }
}
