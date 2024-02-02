use anyhow::{anyhow, Result};
use ash::extensions::khr;
use ash::util::Align;
use ash::vk;
use presentation::Presentation;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use std::ffi::CStr;
use std::path::Path;

mod presentation;

const NAME: &[u8] = env!("CARGO_PKG_NAME").as_bytes();
const VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
const API_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
const CLEAR_COLOR: [ash::vk::ClearValue; 2] = [
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
const SHADERS_PATH: &str = "data\\shaders\\";

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

pub struct Engine {
    pub window: Window,
    instance: ash::Instance,
    _entry: ash::Entry,
    surface_loader: khr::Surface,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    present_queue: vk::Queue,
    presentation: Presentation,
    _command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    command_buffer_fences: [vk::Fence; 2],
    render_pass: vk::RenderPass,

    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    command_buffer_index: usize,
    outdated_presentation: bool,
    attachment_count: u32, // defined by attachment_description length

    viewports: [vk::Viewport; 1],
    scissors: [vk::Rect2D; 1],
    descriptor_sets: Vec<vk::DescriptorSet>,
    vertex_input_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    index_buffer_data: [u32; 6],
}

impl Engine {
    pub fn new(
        application_version: u32,
        win_title: &str,
        win_init_width: u32,
        win_init_height: u32,
    ) -> (Self, EventLoop<()>) {
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

        // SECTION : Vulkan instance
        let entry = ash::Entry::linked();

        unsafe {
            let app_name = CStr::from_bytes_with_nul_unchecked(NAME);

            let app_info = vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                application_version,
                p_engine_name: app_name.as_ptr(),
                engine_version: VERSION,
                api_version: API_VERSION,
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

            // SECTION : Vulkan surface
            let surface_loader = khr::Surface::new(&entry, &instance);
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .expect("Can't create surface.");

            // SECTION : Pick physical device
            let physical_device = pick_physical_device(&instance, surface, &surface_loader);
            let queue_family_index = get_graphics_family_index(&instance, physical_device);

            let device = create_device(&instance, physical_device, queue_family_index);

            let present_queue = device.get_device_queue(queue_family_index, 0);

            // SECTION : Swapchain
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

            let command_buffer_index = 0;

            // SECTION : Command pool
            let command_pool = create_command_pool(&device, queue_family_index);

            // SECTION : Command buffers
            let command_buffers = allocate_command_buffers(&device, command_pool, 2);

            // SECTION : Semaphores

            // SECTION : Fences
            let command_buffer_fences = [create_fence(&device), create_fence(&device)];

            // SECTION : Render pass
            let render_pass = create_render_pass(&device, presentation.format);

            // SECTION : Shaders
            let vertex_shader = load_shader(&device, "data\\shaders\\triangle.vert.spv");
            let fragment_shader = load_shader(&device, "data\\shaders\\triangle.frag.spv");

            // SECTION : Descriptor set
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
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&descriptor_sizes);

            let descriptor_pool = device
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
            let descriptor_info =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings);

            let desc_set_layouts = [device
                .create_descriptor_set_layout(&descriptor_info, None)
                .unwrap()];

            let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&desc_set_layouts);
            let descriptor_sets: Vec<vk::DescriptorSet> =
                device.allocate_descriptor_sets(&desc_alloc_info).unwrap();

            // SECTION : Pipeline layout
            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
                set_layout_count: desc_set_layouts.len() as u32,
                p_set_layouts: desc_set_layouts.as_ptr(),
                ..Default::default()
            };

            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Can't create Vulkan pipeline layout.");

            // SECTION : Vertex input
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
                    offset: std::mem::size_of::<[f32; 4]>() as u32 * 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: std::mem::size_of::<[f32; 4]>() as u32 * 1,
                },
            ];

            let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&vertex_input_binding_descriptions)
                .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
                .build();

            let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };

            // SECTION : Graphics pipeline
            let stages = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vertex_shader,
                    p_name: CStr::from_bytes_with_nul_unchecked(b"main\0").as_ptr(),
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: fragment_shader,
                    p_name: CStr::from_bytes_with_nul_unchecked(b"main\0").as_ptr(),
                    ..Default::default()
                },
            ];

            let tessellation_state = vk::PipelineTessellationStateCreateInfo {
                ..Default::default()
            };

            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: presentation.surface_caps.current_extent.width as f32,
                height: presentation.surface_caps.current_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: presentation.surface_caps.current_extent,
            }];

            let viewport_state = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                p_viewports: viewports.as_ptr(),
                scissor_count: 1,
                p_scissors: scissors.as_ptr(),
                ..Default::default()
            };

            let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::FILL,
                ..Default::default()
            };

            let multisample_state = vk::PipelineMultisampleStateCreateInfo {
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

            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: 1,
                depth_write_enable: 1,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                front: noop_stencil_state,
                back: noop_stencil_state,
                max_depth_bounds: 1.0,
                ..Default::default()
            };

            let color_blend_attachment_states = vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
                ..Default::default()
            };

            let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
                logic_op: vk::LogicOp::CLEAR,
                attachment_count: 1,
                p_attachments: &color_blend_attachment_states,
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
                p_vertex_input_state: &vertex_input_state_info,
                p_input_assembly_state: &vertex_input_assembly_state_info,
                p_tessellation_state: &tessellation_state,
                p_viewport_state: &viewport_state,
                p_rasterization_state: &rasterization_state,
                p_multisample_state: &multisample_state,
                p_depth_stencil_state: &depth_stencil_state,
                p_color_blend_state: &color_blend_state,
                p_dynamic_state: &dynamic_state,
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            };

            let graphics_pipeline = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .expect("Can't create Vulkan graphics pipeline.")[0];

            // SECTION : Indices
            let index_buffer_data = [0u32, 1, 2, 2, 3, 0];
            let index_buffer_info = vk::BufferCreateInfo {
                size: std::mem::size_of_val(&index_buffer_data) as u64,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let index_buffer = device
                .create_buffer(&index_buffer_info, None)
                .expect("Failed to create index buffer.");
            let index_buffer_memory_req = device.get_buffer_memory_requirements(index_buffer);

            let device_memory_properties =
                instance.get_physical_device_memory_properties(physical_device);
            let index_buffer_memory_index = presentation::find_memorytype_index(
                &index_buffer_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("Unable to find suitable memory type index for index buffer.");
            let index_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: index_buffer_memory_req.size,
                memory_type_index: index_buffer_memory_index,
                ..Default::default()
            };
            let index_buffer_memory = device
                .allocate_memory(&index_allocate_info, None)
                .expect("Failed to allocate index buffer memory.");
            let index_ptr = device
                .map_memory(
                    index_buffer_memory,
                    0,
                    index_buffer_memory_req.size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map index buffer memory.");
            let mut index_slice = Align::new(
                index_ptr,
                std::mem::align_of::<u32>() as u64,
                index_buffer_memory_req.size,
            );
            index_slice.copy_from_slice(&index_buffer_data);
            device.unmap_memory(index_buffer_memory);
            device
                .bind_buffer_memory(index_buffer, index_buffer_memory, 0)
                .expect("Failed to bind index buffer memory.");

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
            let vertex_input_buffer = device
                .create_buffer(&vertex_input_buffer_info, None)
                .expect("Failed to create vertex input buffer.");
            let vertex_input_memory_req =
                device.get_buffer_memory_requirements(vertex_input_buffer);
            let vertex_input_buffer_memory_index = presentation::find_memorytype_index(
                &vertex_input_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("Unable to find suitable memory type index for vertex input buffer.");

            let vertex_buffer_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: vertex_input_memory_req.size,
                memory_type_index: vertex_input_buffer_memory_index,
                ..Default::default()
            };
            let vertex_input_buffer_memory = device
                .allocate_memory(&vertex_buffer_allocate_info, None)
                .expect("Failed to allocate vertex input buffer memory.");
            let vertex_ptr = device
                .map_memory(
                    vertex_input_buffer_memory,
                    0,
                    vertex_input_memory_req.size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map vertex input buffer memory.");
            let mut vertex_slice = Align::new(
                vertex_ptr,
                std::mem::align_of::<Vertex>() as u64,
                vertex_input_memory_req.size,
            );
            vertex_slice.copy_from_slice(&vertices);
            device.unmap_memory(vertex_input_buffer_memory);
            device
                .bind_buffer_memory(vertex_input_buffer, vertex_input_buffer_memory, 0)
                .expect("Failed to bind vertex input buffer memory.");

            // SECTION : Uniforms

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
            let uniform_color_buffer = device
                .create_buffer(&uniform_color_buffer_info, None)
                .unwrap();
            let uniform_color_buffer_memory_req =
                device.get_buffer_memory_requirements(uniform_color_buffer);
            let uniform_color_buffer_memory_index = presentation::find_memorytype_index(
                &uniform_color_buffer_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("Unable to find suitable memorytype for the vertex buffer.");

            let uniform_color_buffer_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: uniform_color_buffer_memory_req.size,
                memory_type_index: uniform_color_buffer_memory_index,
                ..Default::default()
            };
            let uniform_color_buffer_memory = device
                .allocate_memory(&uniform_color_buffer_allocate_info, None)
                .unwrap();
            let uniform_ptr = device
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
            device.unmap_memory(uniform_color_buffer_memory);
            device
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
            let image_buffer = device.create_buffer(&image_buffer_info, None).unwrap();
            let image_buffer_memory_req = device.get_buffer_memory_requirements(image_buffer);
            let image_buffer_memory_index = presentation::find_memorytype_index(
                &image_buffer_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("Unable to find suitable memorytype for the image buffer.");

            let image_buffer_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: image_buffer_memory_req.size,
                memory_type_index: image_buffer_memory_index,
                ..Default::default()
            };
            let image_buffer_memory: vk::DeviceMemory = device
                .allocate_memory(&image_buffer_allocate_info, None)
                .unwrap();
            let image_ptr = device
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
            device.unmap_memory(image_buffer_memory);
            device
                .bind_buffer_memory(image_buffer, image_buffer_memory, 0)
                .unwrap();

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
            let texture_image = device.create_image(&texture_create_info, None).unwrap();
            let texture_memory_req = device.get_image_memory_requirements(texture_image);
            let texture_memory_index = presentation::find_memorytype_index(
                &texture_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let texture_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: texture_memory_req.size,
                memory_type_index: texture_memory_index,
                ..Default::default()
            };
            let texture_memory = device
                .allocate_memory(&texture_allocate_info, None)
                .unwrap();
            device
                .bind_image_memory(texture_image, texture_memory, 0)
                .expect("Unable to bind depth image memory");

            record_submit_commandbuffer(
                &device,
                command_buffers[command_buffer_index],
                presentation.swapchain.setup_commands_reuse_fence,
                present_queue,
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

            let sampler = device.create_sampler(&sampler_info, None).unwrap();

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
            let tex_image_view = device
                .create_image_view(&tex_image_view_info, None)
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
            device.update_descriptor_sets(&write_desc_sets, &[]);

            // SECTION : Engine
            let mut engine = Engine {
                window,
                instance,
                _entry: entry,
                surface_loader,
                physical_device,
                device,
                present_queue,
                presentation,
                _command_pool: command_pool,
                command_buffers,
                command_buffer_fences,
                render_pass,
                pipeline_layout,
                graphics_pipeline,
                command_buffer_index,
                outdated_presentation: false,
                attachment_count: 2,
                viewports,
                scissors,
                descriptor_sets,
                vertex_input_buffer,
                index_buffer,
                index_buffer_data,
            };

            // SECTION : Create framebuffers
            engine.create_framebuffers();

            (engine, event_loop)
        }
    }

    fn create_framebuffers(&mut self) {
        for image_view in &self.presentation.swapchain.image_views {
            let framebuffer = create_framebuffer(
                &self.device,
                *image_view,
                self.presentation.swapchain.depth.image_view,
                self.render_pass,
                self.presentation.surface_caps.current_extent.width,
                self.presentation.surface_caps.current_extent.height,
                self.attachment_count,
            );
            self.presentation.framebuffers.push(framebuffer);
        }
    }

    #[inline]
    pub unsafe fn update_presentation(&mut self) {
        self.device.device_wait_idle().unwrap();

        let width = self.window.inner_size().width;
        let height = self.window.inner_size().height;

        if width <= 0 || height <= 0 {
            return;
        }

        println!("updating presentation to {:?} x {:?}", width, height);

        // SECTION : Cleanup depth resources
        self.device
            .destroy_image_view(self.presentation.swapchain.depth.image_view, None);
        self.device
            .destroy_image(self.presentation.swapchain.depth.image, None);

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
            self.presentation.surface_khr,
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

            record_submit_commandbuffer(
                &self.device,
                self.command_buffers[self.command_buffer_index],
                self.presentation.swapchain.draw_commands_reuse_fence,
                self.present_queue,
                &[vk::PipelineStageFlags::BOTTOM_OF_PIPE],
                &[self.presentation.image_available_semaphore],
                &[self.presentation.render_finished_semaphore],
                |device, draw_command_buffer| {
                    device.cmd_begin_render_pass(
                        draw_command_buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );
                    device.cmd_bind_descriptor_sets(
                        draw_command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &self.descriptor_sets,
                        &[],
                    );
                    device.cmd_bind_pipeline(
                        draw_command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.graphics_pipeline,
                    );

                    device.cmd_set_viewport(draw_command_buffer, 0, &self.viewports);
                    device.cmd_set_scissor(draw_command_buffer, 0, &self.scissors);
                    device.cmd_bind_vertex_buffers(
                        draw_command_buffer,
                        0,
                        &[self.vertex_input_buffer],
                        &[0],
                    );
                    device.cmd_bind_index_buffer(
                        draw_command_buffer,
                        self.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.cmd_draw_indexed(
                        draw_command_buffer,
                        self.index_buffer_data.len() as u32,
                        1,
                        0,
                        0,
                        1,
                    );
                    // Or draw without the index buffer
                    // device.cmd_draw(draw_command_buffer, 3, 1, 0, 0);

                    device.cmd_end_render_pass(draw_command_buffer);
                },
            );

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
                .queue_present(self.present_queue, &present_info)
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

        instance
            .create_device(physical_device, &create_info, None)
            .expect("Can't create Vulkan device.")
    }
}

pub fn create_framebuffer(
    device: &ash::Device,
    image_view: vk::ImageView,
    depth_image_view: vk::ImageView,
    render_pass: vk::RenderPass,
    width: u32,
    height: u32,
    attachment_count: u32,
) -> vk::Framebuffer {
    let attachments = [image_view, depth_image_view];

    unsafe {
        let framebuffer_create_info = vk::FramebufferCreateInfo {
            render_pass,
            attachment_count,
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

        device
            .create_command_pool(&create_info, None)
            .expect("Can't create Vulkan command pool.")
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

        device
            .allocate_command_buffers(&allocate_info)
            .expect("Can't allocate Vulkan command buffers.")
    }
}

pub fn create_semaphore(device: &ash::Device) -> vk::Semaphore {
    unsafe {
        let create_info = vk::SemaphoreCreateInfo::default();
        device
            .create_semaphore(&create_info, None)
            .expect("Can't create Vulkan semaphore.")
    }
}

pub fn create_fence(device: &ash::Device) -> vk::Fence {
    unsafe {
        let create_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        device
            .create_fence(&create_info, None)
            .expect("Can't create Vulkan fence.")
    }
}

pub fn create_render_pass(device: &ash::Device, swapchain_format: vk::Format) -> vk::RenderPass {
    unsafe {
        let attachment_description = [
            vk::AttachmentDescription {
                format: swapchain_format,
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

        let create_info = vk::RenderPassCreateInfo {
            attachment_count: attachment_description.len() as u32,
            p_attachments: attachment_description.as_ptr(),
            subpass_count: subpass_description.len() as u32,
            p_subpasses: subpass_description.as_ptr(),
            dependency_count: dependencies.len() as u32,
            p_dependencies: dependencies.as_ptr(),
            ..Default::default()
        };

        device
            .create_render_pass(&create_info, None)
            .expect("Can't create Vulkan render pass.")
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

        device
            .create_shader_module(&shader_module_create_info, None)
            .expect("Can't create Vulkan shader module.")
    }
}

pub fn record_submit_commandbuffer<F: FnOnce(&ash::Device, vk::CommandBuffer)>(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    unsafe {
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

        let command_buffers = vec![command_buffer];

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
}

pub(crate) fn compile_shaders() -> Result<()> {
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
        .arg(format!("{}/triangle.frag", SHADERS_PATH))
        .arg("-o")
        .arg(format!("{}/triangle.frag.spv", SHADERS_PATH))
        .output()
        .expect("Failed to execute glslc.exe for vertex shader");

    let output_frag = std::process::Command::new("glslc.exe")
        .arg(format!("{}/triangle.vert", SHADERS_PATH))
        .arg("-o")
        .arg(format!("{}/triangle.vert.spv", SHADERS_PATH))
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
