use super::{api, utils, Network};
use megenginelite_sys::*;
use std::path::Path;

pub fn default_config() -> LiteConfig {
    unsafe { *api().default_config() }
}

fn default_io() -> LiteNetworkIO {
    unsafe { *api().default_network_io() }
}

pub struct NetworkIO<'a> {
    pub inputs: &'a [LiteIO],
    pub outputs: &'a [LiteIO],
}

impl<'a> Into<LiteNetworkIO> for NetworkIO<'a> {
    fn into(self) -> LiteNetworkIO {
        LiteNetworkIO {
            inputs: self.inputs.as_ptr() as *mut LiteIO,
            outputs: self.outputs.as_ptr() as *mut LiteIO,
            input_size: self.inputs.len() as u64,
            output_size: self.outputs.len() as u64,
        }
    }
}

pub struct NetworkBuilder<'a> {
    pub(super) config: Option<LiteConfig>,
    pub(super) io: Option<LiteNetworkIO>,
    pub(super) option_setting: Vec<Box<dyn FnOnce(LiteNetwork)>>,
    pub(super) phantom: std::marker::PhantomData<&'a Network>,
}

impl<'a> NetworkBuilder<'a> {
    /// Set the configration to create the network
    pub fn config(mut self, config: LiteConfig) -> NetworkBuilder<'a> {
        self.config = Some(config);
        self
    }

    /// Set the configration io to create the network
    pub fn io(mut self, io: NetworkIO<'a>) -> NetworkBuilder<'a> {
        self.io = Some(io.into());
        self
    }

    /// Set cpu default mode when device is CPU, in some low computation
    /// device or single core device, this mode will get good performace
    pub fn cpu_inplace(mut self) -> NetworkBuilder<'a> {
        self.option_setting.push(Box::new(|net| unsafe {
            api().LITE_set_cpu_inplace_mode(net);
        }));
        self
    }

    /// Enable tensorrt
    pub fn tensorrt(mut self) -> NetworkBuilder<'a> {
        self.option_setting.push(Box::new(|net| unsafe {
            api().LITE_use_tensorrt(net);
        }));
        self
    }

    /// When device is CPU, this interface will set the to be loaded model
    /// run in multi thread mode with the given thread number.
    pub fn threads_number(mut self, nr_threads: u64) -> NetworkBuilder<'a> {
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_set_cpu_threads_number(net, nr_threads);
        }));
        self
    }

    /// Set device id, default device id = 0
    pub fn dev_id(mut self, dev_id: i32) -> NetworkBuilder<'a> {
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_set_device_id(net, dev_id);
        }));
        self
    }

    /// Set stream id, default stream id = 0
    pub fn stream_id(mut self, stream_id: i32) -> NetworkBuilder<'a> {
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_set_stream_id(net, stream_id);
        }));
        self
    }

    /// Set opr algorithm selection strategy in the network
    pub fn algo_policy(mut self, strategy: LiteAlgoSelectStrategy) -> NetworkBuilder<'a> {
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_set_network_algo_policy(net, strategy);
        }));
        self
    }

    /// Set opr algorithm selection strategy in the network
    pub fn fastrun_config(
        mut self,
        shared_batch_size: u32,
        binary_equal_between_batch: i32,
    ) -> NetworkBuilder<'a> {
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_set_network_algo_fastrun_config(
                net,
                shared_batch_size,
                binary_equal_between_batch,
            );
        }));
        self
    }

    /// Set workspace_limit for oprs with multiple algorithms, set workspace limit can save memory
    /// but may influence the performance
    pub fn workspace_limit(mut self, workspace_limit: u64) -> NetworkBuilder<'a> {
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_set_network_algo_workspace_limit(net, workspace_limit);
        }));
        self
    }

    /// Enable profile the network, a JSON format file will be generated
    pub fn profile_performance(mut self, path: impl AsRef<Path>) -> NetworkBuilder<'a> {
        let path_str_c = utils::path_to_cstr(path.as_ref());
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_enable_profile_performance(net, path_str_c.as_ptr());
        }));
        self
    }

    /// Dump input/output values of all internal variables to output file
    /// in text format
    pub fn io_txt_dump(mut self, path: impl AsRef<Path>) -> NetworkBuilder<'a> {
        let path_str_c = utils::path_to_cstr(path.as_ref());
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_enable_io_txt_dump(net, path_str_c.as_ptr());
        }));
        self
    }

    /// Dump input/output values of all internal variables to output
    /// directory, in binary format
    pub fn io_bin_dump(mut self, path: impl AsRef<Path>) -> NetworkBuilder<'a> {
        let path_str_c = utils::path_to_cstr(path.as_ref());
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_enable_io_bin_dump(net, path_str_c.as_ptr());
        }));
        self
    }

    /// Share runtime memory with `net`
    pub fn share_runtime_memroy(mut self, net: &'a Network) -> NetworkBuilder<'a> {
        let raw_net = net.inner;
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_share_runtime_memroy(net, raw_net);
        }));
        self
    }

    /// Share weights with `net`
    pub fn share_weights_with(mut self, net: &'a Network) -> NetworkBuilder<'a> {
        let raw_net = net.inner;
        self.option_setting.push(Box::new(move |net| unsafe {
            api().LITE_shared_weight_with_network(net, raw_net);
        }));
        self
    }

    /// Load the model to network form given path
    pub fn build(self, path: impl AsRef<Path>) -> Network {
        let path_str_c = utils::path_to_cstr(path.as_ref());
        let config = self.config.unwrap_or(default_config());
        let io = self.io.unwrap_or(default_io());

        let mut net = std::ptr::null_mut();
        unsafe { api().LITE_make_network(&mut net, config, io) };
        for f in self.option_setting.into_iter() {
            f(net);
        }
        unsafe { api().LITE_load_model_from_path(net, path_str_c.as_ptr()) };

        Network { inner: net }
    }

    /// Load the model to network form memory
    pub fn build_from_memory(self, mem: &mut [u8]) -> Network {
        let config = self.config.unwrap_or(default_config());
        let io = self.io.unwrap_or(default_io());

        let mut net = std::ptr::null_mut();
        unsafe { api().LITE_make_network(&mut net, config, io) };
        for f in self.option_setting.into_iter() {
            f(net);
        }
        unsafe { api().LITE_load_model_from_mem(net, mem.as_ptr() as *mut _, mem.len() as u64) };

        Network { inner: net }
    }
}
