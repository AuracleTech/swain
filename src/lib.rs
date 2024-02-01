use ash::extensions::khr;
use ash::vk;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use std::ffi::CStr;

const NAME: &[u8] = env!("CARGO_PKG_NAME").as_bytes();
const VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
const CLEAR_COLOR: [ash::vk::ClearValue; 1] = [vk::ClearValue {
    color: vk::ClearColorValue {
        float32: [1.0, 1.0, 1.0, 1.0],
    },
}];

pub struct Presentation {
    surface: vk::SurfaceKHR,
    surface_caps: vk::SurfaceCapabilitiesKHR,
    format: vk::Format,
    queue_family_index: u32,
    swapchain_loader: khr::Swapchain,
    swapchain: Swapchain,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Presentation {
    pub fn new(
        surface_loader: &khr::Surface,
        surface: vk::SurfaceKHR,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        queue_family_index: u32,
        image_available_semaphore: vk::Semaphore,
        render_finished_semaphore: vk::Semaphore,
        width: u32,
        height: u32,
    ) -> Self {
        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .expect("Can't get Vulkan surface capabilities.")
        };
        let format = get_swapchain_format(physical_device, surface, &surface_loader);
        let swapchain_loader = khr::Swapchain::new(&instance, &device);

        let swapchain = create_swapchain(
            &swapchain_loader,
            surface,
            &device,
            &surface_caps,
            format,
            queue_family_index,
            vk::SwapchainKHR::null(),
            width,
            height,
        );

        Presentation {
            surface,
            surface_caps,
            format,
            queue_family_index,
            swapchain_loader,
            swapchain,
            image_available_semaphore,
            render_finished_semaphore,
            framebuffers: Vec::new(),
        }
    }
}

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

pub struct Engine {
    pub window: Window,
    instance: ash::Instance,
    _entry: ash::Entry,
    surface_loader: khr::Surface,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    presentation: Presentation,
    _command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    command_buffer_fences: [vk::Fence; 2],
    render_pass: vk::RenderPass,

    _pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    command_buffer_index: usize,
    outdated_presentation: bool,
}

impl Engine {
    pub fn new(
        application_version: u32,
        win_title: &str,
        win_init_width: u32,
        win_init_height: u32,
    ) -> (Self, EventLoop<()>) {
        // SECTION : Create window
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(win_title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                win_init_width,
                win_init_height,
            ))
            .build(&event_loop)
            .unwrap();

        // SECTION : Create Vulkan instance
        let entry = ash::Entry::linked();

        unsafe {
            let app_name = CStr::from_bytes_with_nul_unchecked(NAME);

            let app_info = vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                application_version,
                p_engine_name: app_name.as_ptr(),
                engine_version: VERSION,
                api_version: vk::make_api_version(0, 1, 0, 0),
                ..Default::default()
            };

            #[cfg(debug_assertions)]
            let validation_layers =
                [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()];

            #[cfg(not(debug_assertions))]
            let validation_layers = [];

            let extensions = [
                khr::Surface::name().as_ptr(),
                khr::Win32Surface::name().as_ptr(),
            ];

            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_layer_count: validation_layers.len() as u32,
                pp_enabled_layer_names: validation_layers.as_ptr(),
                enabled_extension_count: extensions.len() as u32,
                pp_enabled_extension_names: extensions.as_ptr(),
                ..Default::default()
            };

            let instance = entry
                .create_instance(&create_info, None)
                .expect("Can't create Vulkan instance.");

            // SECTION : Create Vulkan surface
            let surface_loader = khr::Surface::new(&entry, &instance);
            let surface = ash_window::create_surface(&entry, &instance, &window, None)
                .expect("Can't create surface.");

            // SECTION : Pick physical device
            let physical_device = pick_physical_device(&instance, surface, &surface_loader);
            let queue_family_index = get_graphics_family_index(&instance, physical_device);

            let device = create_device(&instance, physical_device, queue_family_index);

            let graphics_queue = device.get_device_queue(queue_family_index, 0);

            // SECTION : Create swapchain
            let presentation = Presentation::new(
                &surface_loader,
                surface,
                &instance,
                physical_device,
                &device,
                queue_family_index,
                create_semaphore(&device),
                create_semaphore(&device),
                window.inner_size().width,
                window.inner_size().height,
            );

            // SECTION : Create command pool
            let command_pool = create_command_pool(&device, queue_family_index);

            // SECTION : Create command buffers
            let command_buffers = allocate_command_buffers(&device, command_pool, 2);

            // SECTION : Create semaphores

            // SECTION : Create fences
            let command_buffer_fences = [create_fence(&device), create_fence(&device)];

            // SECTION : Create render pass
            let render_pass = create_render_pass(&device, presentation.format);

            // SECTION : Create shaders
            let vertex_shader = load_shader(&device, "data\\shaders\\triangle.vert.spv");
            let fragment_shader = load_shader(&device, "data\\shaders\\triangle.frag.spv");

            // SECTION : Create pipeline layout
            let pipeline_layout = create_pipeline_layout(&device);

            // SECTION : Create graphics pipeline
            let graphics_pipeline = create_graphics_pipeline(
                &device,
                vertex_shader,
                fragment_shader,
                pipeline_layout,
                render_pass,
            );

            // SECTION : Create engine
            let mut engine = Engine {
                window,
                instance,
                _entry: entry,
                surface_loader,
                physical_device,
                device,
                graphics_queue,
                presentation,
                _command_pool: command_pool,
                command_buffers,
                command_buffer_fences,
                render_pass,
                _pipeline_layout: pipeline_layout,
                graphics_pipeline,
                command_buffer_index: 0,
                outdated_presentation: false,
            };

            // SECTION : Create framebuffers
            engine.create_framebuffers();

            (engine, event_loop)
        }
    }

    fn create_framebuffers(&mut self) {
        for image_view in &self.presentation.swapchain.image_views {
            self.presentation.framebuffers.push(create_framebuffer(
                &self.device,
                *image_view,
                self.render_pass,
                self.presentation.surface_caps.current_extent.width,
                self.presentation.surface_caps.current_extent.height,
            ));
        }
    }

    pub unsafe fn update_presentation(&mut self) {
        self.device.device_wait_idle().unwrap();

        let width = self.window.inner_size().width;
        let height = self.window.inner_size().height;

        if width <= 0 || height <= 0 {
            return;
        }

        println!("updating presentation to {:?} x {:?}", width, height);

        // SECTION : Destroy old framebuffers
        for image_view in &self.presentation.swapchain.image_views {
            self.device.destroy_image_view(*image_view, None);
        }

        // SECTION : Destroy old swapchain
        self.presentation
            .swapchain_loader
            .destroy_swapchain(self.presentation.swapchain.swapchain, None);

        // SECTION : Create new presentation
        self.presentation = Presentation::new(
            &self.surface_loader,
            self.presentation.surface,
            &self.instance,
            self.physical_device,
            &self.device,
            self.presentation.queue_family_index,
            self.presentation.image_available_semaphore,
            self.presentation.render_finished_semaphore,
            width,
            height,
        );

        // SECTION : Create new framebuffers
        self.create_framebuffers();

        self.outdated_presentation = false;
    }

    pub fn draw(&mut self) {
        let width = self.window.inner_size().width;
        let height = self.window.inner_size().height;

        if width <= 0 || height <= 0 {
            return;
        }

        if self.outdated_presentation {
            unsafe {
                self.update_presentation();
            }
        }

        unsafe {
            // SECTION : Wait and reset fences
            self.device
                .wait_for_fences(
                    &[self.command_buffer_fences[self.command_buffer_index]],
                    true,
                    std::u64::MAX,
                )
                .unwrap();
            self.device
                .reset_fences(&[self.command_buffer_fences[self.command_buffer_index]])
                .unwrap();

            // SECTION : Acquire next image
            let rendered_image_index = loop {
                match self.presentation.swapchain_loader.acquire_next_image(
                    self.presentation.swapchain.swapchain,
                    std::u64::MAX,
                    self.presentation.image_available_semaphore,
                    vk::Fence::null(),
                ) {
                    Ok((index, _)) => break index,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        self.outdated_presentation = true;
                        return self.draw();
                    }
                    Err(error) => panic!("Can't acquire next image: {:?}", error),
                }
            };

            // SECTION : Begin command buffer
            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };

            self.device
                .begin_command_buffer(
                    self.command_buffers[self.command_buffer_index],
                    &command_buffer_begin_info,
                )
                .expect("Can't begin Vulkan command buffer.");

            // SECTION : Begin barrier
            let render_begin_barrier = vk::ImageMemoryBarrier {
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.presentation.swapchain.images[rendered_image_index as usize],
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };

            self.device.cmd_pipeline_barrier(
                self.command_buffers[self.command_buffer_index],
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[render_begin_barrier],
            );

            // SECTION : Begin render pass
            let render_pass_begin_info = vk::RenderPassBeginInfo {
                render_pass: self.render_pass,
                framebuffer: self.presentation.framebuffers[rendered_image_index as usize],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.presentation.surface_caps.current_extent,
                },
                clear_value_count: CLEAR_COLOR.len() as u32,
                p_clear_values: CLEAR_COLOR.as_ptr(),
                ..Default::default()
            };

            self.device.cmd_begin_render_pass(
                self.command_buffers[self.command_buffer_index],
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            // SECTION : Viewport and Scissor
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            };

            self.device.cmd_set_viewport(
                self.command_buffers[self.command_buffer_index],
                0,
                &[viewport],
            );
            self.device.cmd_set_scissor(
                self.command_buffers[self.command_buffer_index],
                0,
                &[scissor],
            );

            // SECTION : Bind pipeline
            self.device.cmd_bind_pipeline(
                self.command_buffers[self.command_buffer_index],
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );

            // SECTION : Draw
            let vertex_count = 3;
            let instance_count = 1;
            let first_vertex = 0;
            let first_instance = 0;
            self.device.cmd_draw(
                self.command_buffers[self.command_buffer_index],
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );

            // SECTION : End render pass
            self.device
                .cmd_end_render_pass(self.command_buffers[self.command_buffer_index]);

            // SECTION : End barrier
            let render_end_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.presentation.swapchain.images[rendered_image_index as usize],
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier(
                self.command_buffers[self.command_buffer_index],
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[render_end_barrier],
            );

            // SECTION : End command buffer
            self.device
                .end_command_buffer(self.command_buffers[self.command_buffer_index])
                .expect("Can't end Vulkan command buffer.");

            // SECTION : Submit and present
            let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &self.presentation.image_available_semaphore,
                p_wait_dst_stage_mask: wait_dst_stage_mask.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &self.command_buffers[self.command_buffer_index],
                signal_semaphore_count: 1,
                p_signal_semaphores: &self.presentation.render_finished_semaphore,
                ..Default::default()
            };

            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[submit_info],
                    self.command_buffer_fences[self.command_buffer_index],
                )
                .expect("Can't submit Vulkan queue.");

            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: &self.presentation.render_finished_semaphore,
                swapchain_count: 1,
                p_swapchains: &self.presentation.swapchain.swapchain,
                p_image_indices: &rendered_image_index,
                ..Default::default()
            };

            match self
                .presentation
                .swapchain_loader
                .queue_present(self.graphics_queue, &present_info)
            {
                Ok(_) => {
                    self.command_buffer_index = (self.command_buffer_index + 1) % 2;
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.outdated_presentation = true;
                    return self.draw();
                }
                Err(error) => panic!("Can't present queue: {:?}", error),
            }
        }
    }
}

fn get_surface_composite_alpha(
    surface_caps: &vk::SurfaceCapabilitiesKHR,
) -> vk::CompositeAlphaFlagsKHR {
    if surface_caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
    {
        return vk::CompositeAlphaFlagsKHR::OPAQUE;
    } else if surface_caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
    {
        return vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED;
    } else if surface_caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
    {
        return vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED;
    } else {
        return vk::CompositeAlphaFlagsKHR::INHERIT;
    }
}

pub fn get_graphics_family_index(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> u32 {
    unsafe {
        let queue_family_properties =
            instance.get_physical_device_queue_family_properties(physical_device);

        for i in 0..queue_family_properties.len() {
            if queue_family_properties[i]
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
            {
                return i as u32;
            }
        }

        vk::QUEUE_FAMILY_IGNORED
    }
}

pub fn pick_physical_device(
    instance: &ash::Instance,
    surface: vk::SurfaceKHR,
    surface_loader: &khr::Surface,
) -> vk::PhysicalDevice {
    unsafe {
        let physical_devices = instance
            .enumerate_physical_devices()
            .expect("Can't enumerate Vulkan physical devices.");

        let mut discrete = vk::PhysicalDevice::null();
        let mut fallback = vk::PhysicalDevice::null();
        for physical_device in physical_devices {
            let properties = instance.get_physical_device_properties(physical_device);

            let graphics_family_index = get_graphics_family_index(instance, physical_device);
            if graphics_family_index == vk::QUEUE_FAMILY_IGNORED {
                continue;
            }

            let supports_presentation = surface_loader
                .get_physical_device_surface_support(
                    physical_device,
                    graphics_family_index,
                    surface,
                )
                .expect("Can't get physical device surface support info.");
            if !supports_presentation {
                continue;
            }

            if discrete == vk::PhysicalDevice::null()
                && properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            {
                discrete = physical_device;
            } else if fallback == vk::PhysicalDevice::null() {
                fallback = physical_device;
            }
        }

        let physical_device = if discrete != vk::PhysicalDevice::null() {
            discrete
        } else {
            fallback
        };
        if physical_device != vk::PhysicalDevice::null() {
            let properties = instance.get_physical_device_properties(physical_device);
            println!(
                "Selected GPU: {}",
                CStr::from_ptr(properties.device_name.as_ptr())
                    .to_str()
                    .unwrap()
            );
        } else {
            panic!("Can't pick GPU.")
        }

        physical_device
    }
}

pub fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    family_index: u32,
) -> ash::Device {
    unsafe {
        let queue_priorities = [1.0];

        let queue_create_infos = [vk::DeviceQueueCreateInfo {
            queue_family_index: family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        let extensions = [khr::Swapchain::name().as_ptr()];

        let create_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };

        let device = instance
            .create_device(physical_device, &create_info, None)
            .expect("Can't create Vulkan device.");

        device
    }
}

pub fn get_swapchain_format(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &khr::Surface,
) -> vk::Format {
    unsafe {
        let surface_formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("Can't get physical device surface formats.");

        if surface_formats.len() == 1 && surface_formats[0].format == vk::Format::UNDEFINED {
            return vk::Format::R8G8B8A8_UNORM;
        } else {
            for format in &surface_formats {
                if format.format == vk::Format::R8G8B8A8_UNORM
                    || format.format == vk::Format::B8G8R8A8_UNORM
                {
                    return format.format;
                }
            }
        }

        surface_formats[0].format
    }
}

fn create_swapchain(
    swapchain_loader: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    device: &ash::Device,
    surface_caps: &vk::SurfaceCapabilitiesKHR,
    format: vk::Format,
    family_index: u32,
    old_swapchain: vk::SwapchainKHR,
    width: u32,
    height: u32,
) -> Swapchain {
    unsafe {
        let composite_alpha = get_surface_composite_alpha(&surface_caps);

        let create_info = vk::SwapchainCreateInfoKHR {
            surface: surface,
            min_image_count: std::cmp::max(2, surface_caps.min_image_count),
            image_format: format,
            image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            image_extent: vk::Extent2D { width, height },
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            queue_family_index_count: 1,
            p_queue_family_indices: &family_index,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: composite_alpha,
            present_mode: vk::PresentModeKHR::FIFO,
            old_swapchain: old_swapchain,
            ..Default::default()
        };

        let swapchain = swapchain_loader
            .create_swapchain(&create_info, None)
            .expect("Can't create Vulkan swapchain.");

        let images = get_swapchain_images(swapchain, &swapchain_loader);

        let mut image_views: Vec<vk::ImageView> = Vec::with_capacity(images.len());
        for image in &images {
            let image_view = create_image_view(device, *image, format);

            image_views.push(image_view);
        }

        Swapchain {
            swapchain,
            images,
            image_views,
        }
    }
}

fn get_swapchain_images(
    swapchain: vk::SwapchainKHR,
    swapchain_loader: &khr::Swapchain,
) -> Vec<vk::Image> {
    unsafe {
        let swapchain_images = swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("Can't get Vulkan swapchain images.");

        swapchain_images
    }
}

fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    swapchain_format: vk::Format,
) -> vk::ImageView {
    unsafe {
        let image_view_create_info = vk::ImageViewCreateInfo {
            image: image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: swapchain_format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let image_view = device
            .create_image_view(&image_view_create_info, None)
            .expect("Can't create Vulkan image view.");

        image_view
    }
}

pub fn create_framebuffer(
    device: &ash::Device,
    image_view: vk::ImageView,
    render_pass: vk::RenderPass,
    width: u32,
    height: u32,
) -> vk::Framebuffer {
    let attachments = [image_view];

    unsafe {
        let framebuffer_create_info = vk::FramebufferCreateInfo {
            render_pass: render_pass,
            attachment_count: 1,
            p_attachments: attachments.as_ptr(),
            width,
            height,
            layers: 1,
            ..Default::default()
        };

        device
            .create_framebuffer(&framebuffer_create_info, None)
            .expect("Can't create Vulkan framebuffer.")
    }
}

pub fn create_command_pool(device: &ash::Device, family_index: u32) -> vk::CommandPool {
    unsafe {
        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT
                | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: family_index,
            ..Default::default()
        };

        let command_pool = device
            .create_command_pool(&create_info, None)
            .expect("Can't create Vulkan command pool.");

        command_pool
    }
}

pub fn allocate_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Vec<vk::CommandBuffer> {
    unsafe {
        let allocate_info = vk::CommandBufferAllocateInfo {
            command_pool: command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: count,
            ..Default::default()
        };

        let command_buffers = device
            .allocate_command_buffers(&allocate_info)
            .expect("Can't allocate Vulkan command buffers.");

        command_buffers
    }
}

pub fn create_semaphore(device: &ash::Device) -> vk::Semaphore {
    unsafe {
        let create_info = vk::SemaphoreCreateInfo::default();
        let semaphore = device
            .create_semaphore(&create_info, None)
            .expect("Can't create Vulkan semaphore.");

        semaphore
    }
}

pub fn create_fence(device: &ash::Device) -> vk::Fence {
    unsafe {
        let create_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let fence = device
            .create_fence(&create_info, None)
            .expect("Can't create Vulkan fence.");

        fence
    }
}

pub fn create_render_pass(device: &ash::Device, swapchain_format: vk::Format) -> vk::RenderPass {
    unsafe {
        let attachment_description = [vk::AttachmentDescription {
            format: swapchain_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        }];

        let color_attachment = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpass_description = [vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: color_attachment.len() as u32,
            p_color_attachments: color_attachment.as_ptr(),
            ..Default::default()
        }];

        let create_info = vk::RenderPassCreateInfo {
            attachment_count: attachment_description.len() as u32,
            p_attachments: attachment_description.as_ptr(),
            subpass_count: subpass_description.len() as u32,
            p_subpasses: subpass_description.as_ptr(),
            ..Default::default()
        };

        let render_pass = device
            .create_render_pass(&create_info, None)
            .expect("Can't create Vulkan render pass.");

        render_pass
    }
}

pub fn load_shader(device: &ash::Device, path: &str) -> vk::ShaderModule {
    unsafe {
        let vertex_shader_code = std::fs::read(path).expect("Can't read Vulkan spv file.");

        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: vertex_shader_code.len(),
            p_code: vertex_shader_code.as_ptr() as *const u32,
            ..Default::default()
        };

        let shader_module = device
            .create_shader_module(&shader_module_create_info, None)
            .expect("Can't create Vulkan shader module.");

        shader_module
    }
}

pub fn create_pipeline_layout(device: &ash::Device) -> vk::PipelineLayout {
    unsafe {
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            ..Default::default()
        };

        let pipeline_layout = device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .expect("Can't create Vulkan pipeline layout.");

        pipeline_layout
    }
}

pub fn create_graphics_pipeline(
    device: &ash::Device,
    vs: vk::ShaderModule,
    fs: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> vk::Pipeline {
    unsafe {
        let stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vs,
                p_name: CStr::from_bytes_with_nul_unchecked(b"main\0").as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: fs,
                p_name: CStr::from_bytes_with_nul_unchecked(b"main\0").as_ptr(),
                ..Default::default()
            },
        ];

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            ..Default::default()
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let tessellation_state = vk::PipelineTessellationStateCreateInfo {
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            line_width: 1.0,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            ..Default::default()
        };

        let pipeline_color_attachment_blend_state = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
            ..Default::default()
        };

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: 1,
            p_attachments: &pipeline_color_attachment_blend_state,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let create_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_tessellation_state: &tessellation_state,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization_state,
            p_multisample_state: &multisample_state,
            p_depth_stencil_state: &depth_stencil_state,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass: render_pass,
            ..Default::default()
        };

        let graphics_pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .expect("Can't create Vulkan graphics pipeline.");

        graphics_pipeline[0]
    }
}
