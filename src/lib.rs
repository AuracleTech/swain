use anyhow::{anyhow, Result};
use ash::extensions::khr;
use ash::util::Align;
use ash::vk;
use log::info;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::ffi::CStr;
use std::path::Path;
use std::time::{Duration, Instant};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

const NAME: &[u8] = env!("CARGO_PKG_NAME").as_bytes();
const VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
const API_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
const SHADERS_PATH: &str = "data/shaders/";
const FPS_TARGET: usize = 15;

lazy_static::lazy_static! {
pub static ref DRAW_TIME_MAX: Duration = Duration::from_secs_f64(1.0 / FPS_TARGET as f64);
}

#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = std::mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}

#[derive(Clone, Debug, Copy)]
struct Vertex {
    pos: [f32; 4],
    uv: [f32; 2],
}

#[derive(Clone, Debug, Copy)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

#[allow(unused)]
pub struct Engine {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    surface_loader: khr::Surface,
    swapchain_loader: khr::Swapchain,
    window: winit::window::Window,

    physical_device: vk::PhysicalDevice,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family_index: u32,
    present_queue: vk::Queue,

    surface_khr: vk::SurfaceKHR,
    surface_format_khr: vk::SurfaceFormatKHR,
    surface_image_format: vk::Format,
    surface_resolution: vk::Extent2D,
    surface_caps: vk::SurfaceCapabilitiesKHR,

    swapchain_khr: vk::SwapchainKHR,
    present_images: Vec<vk::Image>,
    present_image_views: Vec<vk::ImageView>,

    command_pool: vk::CommandPool,
    draw_command_buffer: vk::CommandBuffer,
    setup_command_buffer: vk::CommandBuffer,

    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,

    present_complete_semaphore: vk::Semaphore,
    rendering_complete_semaphore: vk::Semaphore,

    draw_commands_reuse_fence: vk::Fence,
    setup_commands_reuse_fence: vk::Fence,

    pub last_frame_time: Instant,
}

impl Engine {
    pub fn new(
        application_version: u32,
        win_title: &str,
        win_init_width: u32,
        win_init_height: u32,
    ) -> (Self, EventLoop<()>) {
        env_logger::init();

        #[cfg(debug_assertions)]
        compile_shaders().expect("Unable to compile shaders.");

        // SECTION : Window
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(win_title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                win_init_width,
                win_init_height,
            ))
            .build(&event_loop)
            .unwrap();

        let width = window.inner_size().width;
        let height = window.inner_size().height;

        // SECTION : Vulkan instance
        let entry = ash::Entry::linked();

        unsafe {
            #[cfg(debug_assertions)]
            let layer_names =
                [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()];

            #[cfg(not(debug_assertions))]
            let layer_names = [];

            let extension_names = [
                khr::Surface::name().as_ptr(),
                khr::Win32Surface::name().as_ptr(),
            ];

            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                extension_names.push(KhrPortabilityEnumerationFn::NAME.as_ptr());
                extension_names.push(KhrGetPhysicalDeviceProperties2Fn::NAME.as_ptr());
            }

            let app_name = CStr::from_bytes_with_nul_unchecked(NAME);
            let app_info = vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                application_version,
                p_engine_name: app_name.as_ptr(),
                engine_version: VERSION,
                api_version: API_VERSION,
                ..Default::default()
            };

            let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            };

            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_layer_count: layer_names.len() as u32,
                pp_enabled_layer_names: layer_names.as_ptr(),
                enabled_extension_count: extension_names.len() as u32,
                pp_enabled_extension_names: extension_names.as_ptr(),
                flags: create_flags,
                ..Default::default()
            };

            let instance = entry
                .create_instance(&create_info, None)
                .expect("Unable to create Vulkan instance.");

            // SECTION : Vulkan surface
            let surface_khr = ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .unwrap();

            // SECTION : Pick physical device
            let physical_devices = instance.enumerate_physical_devices().unwrap();
            let surface_loader = khr::Surface::new(&entry, &instance);
            let (physical_device, queue_family_index) = physical_devices
                .iter()
                .find_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface_khr,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                })
                .unwrap();
            let queue_family_index = queue_family_index as u32;

            // SECTION : Device
            let device_extension_names_raw = [
                khr::Swapchain::name().as_ptr(),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                KhrPortabilitySubsetFn::NAME.as_ptr(),
            ];
            let enabled_features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };
            let queue_priorities = [1.0];

            let queue_create_infos = [vk::DeviceQueueCreateInfo {
                queue_family_index,
                queue_count: queue_priorities.len() as u32,
                p_queue_priorities: queue_priorities.as_ptr(),
                ..Default::default()
            }];

            let create_info = vk::DeviceCreateInfo {
                queue_create_info_count: queue_create_infos.len() as u32,
                p_queue_create_infos: queue_create_infos.as_ptr(),
                enabled_extension_count: device_extension_names_raw.len() as u32,
                pp_enabled_extension_names: device_extension_names_raw.as_ptr(),
                p_enabled_features: &enabled_features,
                ..Default::default()
            };

            let device = instance
                .create_device(physical_device, &create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index, 0);

            // SECTION : Presentation surface format
            let surfaces_format = surface_loader
                .get_physical_device_surface_formats(physical_device, surface_khr)
                .unwrap();
            let surface_format_khr = surfaces_format[0];
            let surface_image_format = surface_format_khr.format;

            // SECTION : Presentation swapchain
            let surface_caps = surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface_khr)
                .unwrap();

            let mut desired_image_count = surface_caps.min_image_count + 1;
            if surface_caps.max_image_count > 0
                && desired_image_count > surface_caps.max_image_count
            {
                desired_image_count = surface_caps.max_image_count;
            }

            let surface_resolution = match surface_caps.current_extent.width {
                u32::MAX => vk::Extent2D { width, height },
                _ => surface_caps.current_extent,
            };
            let pre_transform = if surface_caps
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_caps.current_transform
            };
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface_khr)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_loader = khr::Swapchain::new(&instance, &device);

            // SECTION : Presentation composite alpha
            let composite_alpha = vk::CompositeAlphaFlagsKHR::OPAQUE;

            let image_color_space = surfaces_format[0].color_space;

            // SECTION : Presentation Chroma & Image Views
            let swapchain_create_info = vk::SwapchainCreateInfoKHR {
                surface: surface_khr,
                min_image_count: desired_image_count,
                image_format: surface_image_format,
                image_color_space,
                image_extent: surface_resolution,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform,
                queue_family_index_count: 1,
                p_queue_family_indices: &queue_family_index,
                composite_alpha,
                present_mode,
                old_swapchain: vk::SwapchainKHR::null(),
                ..Default::default()
            };

            let swapchain_khr = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Can't create Vulkan swapchain.");

            // SECTION : Command pool
            let command_pool_flags = vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;

            let command_pool_create_info = vk::CommandPoolCreateInfo {
                flags: command_pool_flags,
                queue_family_index,
                ..Default::default()
            };

            let command_pool = device
                .create_command_pool(&command_pool_create_info, None)
                .unwrap();

            // SECTION : Command buffers
            let command_buffer_count = 2;
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count,
                ..Default::default()
            };
            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();

            let setup_command_buffer = command_buffers[0];
            let draw_command_buffer = command_buffers[1];

            // SECTION : Presentation swapchain images
            let present_images = swapchain_loader
                .get_swapchain_images(swapchain_khr)
                .unwrap();

            let mut present_image_views = Vec::with_capacity(present_images.len());
            for image in &present_images {
                let image_view_create_info = vk::ImageViewCreateInfo {
                    image: *image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: surface_image_format,
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                };

                let image_view = device
                    .create_image_view(&image_view_create_info, None)
                    .unwrap();

                present_image_views.push(image_view);
            }

            // SECTION : Presentation Depth image & view
            let device_memory_properties =
                instance.get_physical_device_memory_properties(physical_device);
            let depth_format = vk::Format::D16_UNORM;
            let depth_image_create_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: depth_format,
                extent: surface_resolution.into(),
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let depth_image = device.create_image(&depth_image_create_info, None).unwrap();

            let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .unwrap();

            let depth_image_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: depth_image_memory_req.size,
                memory_type_index: depth_image_memory_index,
                ..Default::default()
            };

            let depth_image_memory = device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .unwrap();

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

            let draw_commands_reuse_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");
            let setup_commands_reuse_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");

            //  SECTION : Record setup command buffer
            record_submit_commandbuffer(
                &device,
                setup_command_buffer,
                setup_commands_reuse_fence,
                present_queue,
                &[],
                &[],
                &[],
                |device, setup_command_buffer| {
                    let layout_transition_barriers = [vk::ImageMemoryBarrier {
                        image: depth_image,
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        new_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            level_count: 1,
                            layer_count: 1,
                            ..Default::default()
                        },
                        ..Default::default()
                    }];

                    device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &layout_transition_barriers,
                    );
                },
            );

            // SECTION : Depth image view
            let depth_image_view_info = vk::ImageViewCreateInfo {
                image: depth_image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: depth_format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .expect("Can't create Vulkan image view.");

            // SECTION : Semaphores
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let engine = Self {
                entry,
                instance,
                device,
                queue_family_index,
                physical_device,
                device_memory_properties,
                window,
                surface_loader,
                surface_format_khr,
                surface_image_format,
                present_queue,
                surface_resolution,
                swapchain_loader,
                swapchain_khr,
                surface_caps,
                present_images,
                present_image_views,
                command_pool,
                draw_command_buffer,
                setup_command_buffer,
                depth_image,
                depth_image_view,
                present_complete_semaphore,
                rendering_complete_semaphore,
                draw_commands_reuse_fence,
                setup_commands_reuse_fence,
                surface_khr,
                depth_image_memory,
                last_frame_time: Instant::now(),
            };

            (engine, event_loop)
        }
    }

    pub unsafe fn draw(&mut self) {
        let frame_start_time = Instant::now();
        // SECTION : Render pass
        let attachment_description = [
            vk::AttachmentDescription {
                format: self.surface_image_format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            },
            vk::AttachmentDescription {
                format: vk::Format::D16_UNORM,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
        ];
        let attachment_count = attachment_description.len() as u32;

        let color_attachment = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let subpass_description = [vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: color_attachment.len() as u32,
            p_color_attachments: color_attachment.as_ptr(),
            p_depth_stencil_attachment: &depth_attachment_ref,
            ..Default::default()
        }];

        let render_pass_create_info = vk::RenderPassCreateInfo {
            attachment_count: attachment_description.len() as u32,
            p_attachments: attachment_description.as_ptr(),
            subpass_count: subpass_description.len() as u32,
            p_subpasses: subpass_description.as_ptr(),
            dependency_count: dependencies.len() as u32,
            p_dependencies: dependencies.as_ptr(),
            ..Default::default()
        };

        let render_pass = self
            .device
            .create_render_pass(&render_pass_create_info, None)
            .unwrap();

        // SECTION : Framebuffers
        let mut framebuffers = Vec::with_capacity(self.present_image_views.len());
        for image_view in &self.present_image_views {
            let attachments = [*image_view, self.depth_image_view];
            let p_attachments = attachments.as_ptr();

            let framebuffer_create_info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count,
                p_attachments,
                width: self.surface_resolution.width,
                height: self.surface_resolution.height,
                layers: 1,
                ..Default::default()
            };

            let framebuffer = self
                .device
                .create_framebuffer(&framebuffer_create_info, None)
                .unwrap();

            framebuffers.push(framebuffer);
        }

        // SECTION : Indices
        let index_buffer_data = [0u32, 1, 2, 2, 3, 0];
        let index_buffer_info = vk::BufferCreateInfo {
            size: std::mem::size_of_val(&index_buffer_data) as u64,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let index_buffer = self.device.create_buffer(&index_buffer_info, None).unwrap();
        let index_buffer_memory_req = self.device.get_buffer_memory_requirements(index_buffer);
        let index_buffer_memory_index = find_memorytype_index(
            &index_buffer_memory_req,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for the index buffer.");
        let index_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: index_buffer_memory_req.size,
            memory_type_index: index_buffer_memory_index,
            ..Default::default()
        };
        let index_buffer_memory = self
            .device
            .allocate_memory(&index_allocate_info, None)
            .unwrap();
        let index_ptr = self
            .device
            .map_memory(
                index_buffer_memory,
                0,
                index_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();
        let mut index_slice = Align::new(
            index_ptr,
            std::mem::align_of::<u32>() as u64,
            index_buffer_memory_req.size,
        );
        index_slice.copy_from_slice(&index_buffer_data);
        self.device.unmap_memory(index_buffer_memory);
        self.device
            .bind_buffer_memory(index_buffer, index_buffer_memory, 0)
            .unwrap();

        // SECTION : Vertices
        let vertices = [
            Vertex {
                pos: [-1.0, -1.0, 0.0, 1.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                pos: [-1.0, 1.0, 0.0, 1.0],
                uv: [0.0, 1.0],
            },
            Vertex {
                pos: [1.0, 1.0, 0.0, 1.0],
                uv: [1.0, 1.0],
            },
            Vertex {
                pos: [1.0, -1.0, 0.0, 1.0],
                uv: [1.0, 0.0],
            },
        ];
        let vertex_input_buffer_info = vk::BufferCreateInfo {
            size: std::mem::size_of_val(&vertices) as u64,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let vertex_input_buffer: vk::Buffer = self
            .device
            .create_buffer(&vertex_input_buffer_info, None)
            .unwrap();
        let vertex_input_buffer_memory_req = self
            .device
            .get_buffer_memory_requirements(vertex_input_buffer);
        let vertex_input_buffer_memory_index = find_memorytype_index(
            &vertex_input_buffer_memory_req,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for the vertex buffer.");

        let vertex_buffer_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: vertex_input_buffer_memory_req.size,
            memory_type_index: vertex_input_buffer_memory_index,
            ..Default::default()
        };
        let vertex_input_buffer_memory = self
            .device
            .allocate_memory(&vertex_buffer_allocate_info, None)
            .unwrap();

        let vert_ptr = self
            .device
            .map_memory(
                vertex_input_buffer_memory,
                0,
                vertex_input_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();
        let mut slice = Align::new(
            vert_ptr,
            std::mem::align_of::<Vertex>() as u64,
            vertex_input_buffer_memory_req.size,
        );
        slice.copy_from_slice(&vertices);
        self.device.unmap_memory(vertex_input_buffer_memory);
        self.device
            .bind_buffer_memory(vertex_input_buffer, vertex_input_buffer_memory, 0)
            .unwrap();

        // SECTION : Uniform
        let uniform_color_buffer_data = Vector3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            _pad: 0.0,
        };
        let uniform_color_buffer_info = vk::BufferCreateInfo {
            size: std::mem::size_of_val(&uniform_color_buffer_data) as u64,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let uniform_color_buffer = self
            .device
            .create_buffer(&uniform_color_buffer_info, None)
            .unwrap();
        let uniform_color_buffer_memory_req = self
            .device
            .get_buffer_memory_requirements(uniform_color_buffer);
        let uniform_color_buffer_memory_index = find_memorytype_index(
            &uniform_color_buffer_memory_req,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for the vertex buffer.");

        let uniform_color_buffer_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: uniform_color_buffer_memory_req.size,
            memory_type_index: uniform_color_buffer_memory_index,
            ..Default::default()
        };
        let uniform_color_buffer_memory = self
            .device
            .allocate_memory(&uniform_color_buffer_allocate_info, None)
            .unwrap();
        let uniform_ptr = self
            .device
            .map_memory(
                uniform_color_buffer_memory,
                0,
                uniform_color_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();
        let mut uniform_aligned_slice = Align::new(
            uniform_ptr,
            std::mem::align_of::<Vector3>() as u64,
            uniform_color_buffer_memory_req.size,
        );
        uniform_aligned_slice.copy_from_slice(&[uniform_color_buffer_data]);
        self.device.unmap_memory(uniform_color_buffer_memory);
        self.device
            .bind_buffer_memory(uniform_color_buffer, uniform_color_buffer_memory, 0)
            .unwrap();

        // SECTION : Image
        let image = image::load_from_memory(include_bytes!("../data/image.png"))
            .unwrap()
            .to_rgba8();
        let (width, height) = image.dimensions();
        let image_extent = vk::Extent2D { width, height };
        let image_data = image.into_raw();
        let image_buffer_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<u8>() * image_data.len()) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let image_buffer = self.device.create_buffer(&image_buffer_info, None).unwrap();
        let image_buffer_memory_req = self.device.get_buffer_memory_requirements(image_buffer);
        let image_buffer_memory_index = find_memorytype_index(
            &image_buffer_memory_req,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memorytype for the image buffer.");

        let image_buffer_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: image_buffer_memory_req.size,
            memory_type_index: image_buffer_memory_index,
            ..Default::default()
        };
        let image_buffer_memory = self
            .device
            .allocate_memory(&image_buffer_allocate_info, None)
            .unwrap();
        let image_ptr = self
            .device
            .map_memory(
                image_buffer_memory,
                0,
                image_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();
        let mut image_slice = Align::new(
            image_ptr,
            std::mem::align_of::<u8>() as u64,
            image_buffer_memory_req.size,
        );
        image_slice.copy_from_slice(&image_data);
        self.device.unmap_memory(image_buffer_memory);
        self.device
            .bind_buffer_memory(image_buffer, image_buffer_memory, 0)
            .unwrap();

        // SECTION : Texture
        let texture_create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UNORM,
            extent: image_extent.into(),
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let texture_image = self
            .device
            .create_image(&texture_create_info, None)
            .unwrap();
        let texture_memory_req = self.device.get_image_memory_requirements(texture_image);
        let texture_memory_index = find_memorytype_index(
            &texture_memory_req,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("Unable to find suitable memory index for depth image.");

        let texture_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: texture_memory_req.size,
            memory_type_index: texture_memory_index,
            ..Default::default()
        };
        let texture_memory = self
            .device
            .allocate_memory(&texture_allocate_info, None)
            .unwrap();
        self.device
            .bind_image_memory(texture_image, texture_memory, 0)
            .expect("Unable to bind depth image memory");

        // SECTION : Texture commands
        record_submit_commandbuffer(
            &self.device,
            self.setup_command_buffer,
            self.setup_commands_reuse_fence,
            self.present_queue,
            &[],
            &[],
            &[],
            |device, texture_command_buffer| {
                let texture_barrier = vk::ImageMemoryBarrier {
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    image: texture_image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                device.cmd_pipeline_barrier(
                    texture_command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[texture_barrier],
                );
                let buffer_copy_regions = vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 1,
                        ..Default::default()
                    },
                    image_extent: image_extent.into(),
                    ..Default::default()
                };

                device.cmd_copy_buffer_to_image(
                    texture_command_buffer,
                    image_buffer,
                    texture_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[buffer_copy_regions],
                );
                let texture_barrier_end = vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image: texture_image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                device.cmd_pipeline_barrier(
                    texture_command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[texture_barrier_end],
                );
            },
        );

        // SECTION : Sampler
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_v: vk::SamplerAddressMode::MIRRORED_REPEAT,
            address_mode_w: vk::SamplerAddressMode::MIRRORED_REPEAT,
            max_anisotropy: 1.0,
            border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
            compare_op: vk::CompareOp::NEVER,
            ..Default::default()
        };

        let sampler = self.device.create_sampler(&sampler_info, None).unwrap();

        // SECTION : Texture image view
        let tex_image_view_info = vk::ImageViewCreateInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            format: texture_create_info.format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            image: texture_image,
            ..Default::default()
        };
        let tex_image_view = self
            .device
            .create_image_view(&tex_image_view_info, None)
            .unwrap();

        // SECTION : Descriptor
        let descriptor_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: descriptor_sizes.len() as u32,
            p_pool_sizes: descriptor_sizes.as_ptr(),
            ..Default::default()
        };

        let descriptor_pool = self
            .device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap();
        let desc_layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let descriptor_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: desc_layout_bindings.len() as u32,
            p_bindings: desc_layout_bindings.as_ptr(),
            ..Default::default()
        };

        let desc_set_layouts = [self
            .device
            .create_descriptor_set_layout(&descriptor_info, None)
            .unwrap()];

        let desc_alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: desc_set_layouts.as_ptr(),
            ..Default::default()
        };
        let descriptor_sets = self
            .device
            .allocate_descriptor_sets(&desc_alloc_info)
            .unwrap();

        let uniform_color_buffer_descriptor = vk::DescriptorBufferInfo {
            buffer: uniform_color_buffer,
            offset: 0,
            range: std::mem::size_of_val(&uniform_color_buffer_data) as u64,
        };

        let tex_descriptor = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: tex_image_view,
            sampler,
        };

        let write_desc_sets = [
            vk::WriteDescriptorSet {
                dst_set: descriptor_sets[0],
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_buffer_info: &uniform_color_buffer_descriptor,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_sets[0],
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &tex_descriptor,
                ..Default::default()
            },
        ];
        self.device.update_descriptor_sets(&write_desc_sets, &[]);

        // SECTION : Shaders
        let vertex_shader_code = std::fs::read("data/shaders/vert.spv").unwrap();
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: vertex_shader_code.len(),
            p_code: vertex_shader_code.as_ptr() as *const u32,
            ..Default::default()
        };
        let vertex_shader_module = self
            .device
            .create_shader_module(&shader_module_create_info, None)
            .unwrap();

        let fragment_shader_code = std::fs::read("data/shaders/frag.spv").unwrap();
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: fragment_shader_code.len(),
            p_code: fragment_shader_code.as_ptr() as *const u32,
            ..Default::default()
        };
        let fragment_shader_module = self
            .device
            .create_shader_module(&shader_module_create_info, None)
            .unwrap();

        // SECTION : Pipeline
        let layout_create_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&desc_set_layouts);

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");
        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo {
                module: vertex_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                module: fragment_shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let vertex_input_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Vertex, uv) as u32,
            },
        ];
        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: vertex_input_binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: vertex_input_binding_descriptions.as_ptr(),
            vertex_attribute_description_count: vertex_input_attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: vertex_input_attribute_descriptions.as_ptr(),
            ..Default::default()
        };

        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.surface_resolution.width as f32,
            height: self.surface_resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [self.surface_resolution.into()];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo {
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            ..Default::default()
        };

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            ..Default::default()
        };

        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            ..Default::default()
        };

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            logic_op: vk::LogicOp::CLEAR,
            ..Default::default()
        };

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_state.len() as u32,
            p_dynamic_states: dynamic_state.as_ptr(),
            ..Default::default()
        };

        let graphic_pipeline_infos = vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stage_create_infos.len() as u32,
            p_stages: shader_stage_create_infos.as_ptr(),
            p_vertex_input_state: &vertex_input_state_info,
            p_input_assembly_state: &vertex_input_assembly_state_info,
            p_viewport_state: &viewport_state_info,
            p_rasterization_state: &rasterization_info,
            p_multisample_state: &multisample_state_info,
            p_depth_stencil_state: &depth_state_info,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state_info,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };

        let graphics_pipelines = self
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[graphic_pipeline_infos], None)
            .unwrap();

        let graphic_pipeline = graphics_pipelines[0];

        // SECTION : DRAW
        let width = self.window.inner_size().width;
        let height = self.window.inner_size().height;

        if width == 0 || height == 0 {
            return;
        }

        let (present_index, _) = self
            .swapchain_loader
            .acquire_next_image(
                self.swapchain_khr,
                std::u64::MAX,
                self.present_complete_semaphore,
                vk::Fence::null(),
            )
            .unwrap();
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo {
            render_pass,
            framebuffer: framebuffers[present_index as usize],
            render_area: self.surface_resolution.into(),
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        record_submit_commandbuffer(
            &self.device,
            self.draw_command_buffer,
            self.draw_commands_reuse_fence,
            self.present_queue,
            &[vk::PipelineStageFlags::BOTTOM_OF_PIPE],
            &[self.present_complete_semaphore],
            &[self.rendering_complete_semaphore],
            |device, draw_command_buffer| {
                device.cmd_begin_render_pass(
                    draw_command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_descriptor_sets(
                    draw_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets[..],
                    &[],
                );
                device.cmd_bind_pipeline(
                    draw_command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphic_pipeline,
                );
                device.cmd_set_viewport(draw_command_buffer, 0, &viewports);
                device.cmd_set_scissor(draw_command_buffer, 0, &scissors);
                device.cmd_bind_vertex_buffers(
                    draw_command_buffer,
                    0,
                    &[vertex_input_buffer],
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    draw_command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(
                    draw_command_buffer,
                    index_buffer_data.len() as u32,
                    1,
                    0,
                    0,
                    1,
                );

                device.cmd_end_render_pass(draw_command_buffer);
            },
        );
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.rendering_complete_semaphore,
            swapchain_count: 1,
            p_swapchains: &self.swapchain_khr,
            p_image_indices: &present_index,
            ..Default::default()
        };
        self.swapchain_loader
            .queue_present(self.present_queue, &present_info)
            .unwrap();
        self.last_frame_time = Instant::now();

        // SECTION : Clean up
        self.device.device_wait_idle().unwrap();

        for pipeline in graphics_pipelines {
            self.device.destroy_pipeline(pipeline, None);
        }
        self.device.destroy_pipeline_layout(pipeline_layout, None);
        self.device
            .destroy_shader_module(vertex_shader_module, None);
        self.device
            .destroy_shader_module(fragment_shader_module, None);
        self.device.free_memory(image_buffer_memory, None);
        self.device.destroy_buffer(image_buffer, None);
        self.device.free_memory(texture_memory, None);
        self.device.destroy_image_view(tex_image_view, None);
        self.device.destroy_image(texture_image, None);
        self.device.free_memory(index_buffer_memory, None);
        self.device.destroy_buffer(index_buffer, None);
        self.device.free_memory(uniform_color_buffer_memory, None);
        self.device.destroy_buffer(uniform_color_buffer, None);
        self.device.free_memory(vertex_input_buffer_memory, None);
        self.device.destroy_buffer(vertex_input_buffer, None);
        for &descriptor_set_layout in desc_set_layouts.iter() {
            self.device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
        }
        self.device.destroy_descriptor_pool(descriptor_pool, None);
        self.device.destroy_sampler(sampler, None);
        for framebuffer in framebuffers {
            self.device.destroy_framebuffer(framebuffer, None);
        }
        self.device.destroy_render_pass(render_pass, None);

        // SECTION : Frame counter
        let current_draw_duration = frame_start_time.elapsed();

        info!(
            "draw {:.2} ms | estimated FPS without cap {:.2}",
            current_draw_duration.as_micros() as f64 / 1000.0,
            1_000_000.0 / current_draw_duration.as_micros() as f64
        );
    }
}

unsafe fn record_submit_commandbuffer<F: FnOnce(&ash::Device, vk::CommandBuffer)>(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    device
        .wait_for_fences(&[command_buffer_reuse_fence], true, u64::MAX)
        .expect("Wait for fence failed.");

    device
        .reset_fences(&[command_buffer_reuse_fence])
        .expect("Reset fences failed.");

    device
        .reset_command_buffer(
            command_buffer,
            vk::CommandBufferResetFlags::RELEASE_RESOURCES,
        )
        .expect("Reset command buffer failed.");

    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };

    device
        .begin_command_buffer(command_buffer, &command_buffer_begin_info)
        .expect("Begin commandbuffer");
    f(device, command_buffer);
    device
        .end_command_buffer(command_buffer)
        .expect("End commandbuffer");

    let command_buffers = [command_buffer];

    let submit_info = ash::vk::SubmitInfo {
        wait_semaphore_count: wait_semaphores.len() as u32,
        p_wait_semaphores: wait_semaphores.as_ptr(),
        p_wait_dst_stage_mask: wait_mask.as_ptr(),
        command_buffer_count: command_buffers.len() as u32,
        p_command_buffers: command_buffers.as_ptr(),
        signal_semaphore_count: signal_semaphores.len() as u32,
        p_signal_semaphores: signal_semaphores.as_ptr(),
        ..Default::default()
    };

    device
        .queue_submit(submit_queue, &[submit_info], command_buffer_reuse_fence)
        .expect("queue submit failed.");
}

fn compile_shaders() -> Result<()> {
    if Path::new(SHADERS_PATH).exists() {
        let files = std::fs::read_dir(SHADERS_PATH)?;
        for file in files {
            let file = file?;
            let path = file.path();
            if let Some(ext) = path.extension() {
                if ext == "spv" {
                    std::fs::remove_file(path)?;
                }
            }
        }
    } else {
        std::fs::create_dir(SHADERS_PATH)?;
    }

    let output_vert = std::process::Command::new("glslc.exe")
        .arg(format!("{}code.frag", SHADERS_PATH))
        .arg("-o")
        .arg(format!("{}frag.spv", SHADERS_PATH))
        .output()
        .expect("Failed to execute glslc.exe for vertex shader");

    let output_frag = std::process::Command::new("glslc.exe")
        .arg(format!("{}code.vert", SHADERS_PATH))
        .arg("-o")
        .arg(format!("{}vert.spv", SHADERS_PATH))
        .output()
        .expect("Failed to execute glslc.exe for fragment shader");

    if !(output_vert.status.success() && output_frag.status.success()) {
        Err(anyhow!(
            "Failed to compile shaders:\n{}\n{}",
            String::from_utf8_lossy(&output_vert.stderr),
            String::from_utf8_lossy(&output_frag.stderr)
        ))?;
    }

    Ok(())
}

fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.device
                .destroy_semaphore(self.rendering_complete_semaphore, None);
            self.device
                .destroy_fence(self.draw_commands_reuse_fence, None);
            self.device
                .destroy_fence(self.setup_commands_reuse_fence, None);
            self.device.free_memory(self.depth_image_memory, None);
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            for &image_view in self.present_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain_khr, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface_khr, None);
            self.instance.destroy_instance(None);
        }
    }
}
