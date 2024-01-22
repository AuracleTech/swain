use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use ash::extensions::khr;
use ash::vk;

const WIN_TITLE: &'static str = "Vulkan";
const WIN_INIT_WIDTH: u32 = 1600;
const WIN_INIT_HEIGHT: u32 = 900;

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(WIN_TITLE)
        .with_inner_size(winit::dpi::LogicalSize::new(
            f64::from(WIN_INIT_WIDTH),
            f64::from(WIN_INIT_HEIGHT),
        ))
        .build(&event_loop)
        .unwrap();

    let entry = ash::Entry::linked();

    let instance = swain::create_instance(&entry);

    let surface_loader = khr::Surface::new(&entry, &instance);
    let surface = swain::create_surface(&entry, &instance, &window);

    let physical_device = swain::pick_physical_device(&instance, surface, &surface_loader);
    let graphics_family_index = swain::get_graphics_family_index(&instance, physical_device);

    let device = swain::create_device(&instance, physical_device, graphics_family_index);

    let graphics_queue = swain::get_device_queue(&device, graphics_family_index);

    let surface_caps = swain::get_surface_capabilities(physical_device, surface, &surface_loader);
    let swapchain_format = swain::get_swapchain_format(physical_device, surface, &surface_loader);

    let swapchain_loader = khr::Swapchain::new(&instance, &device);
    let mut swapchain = swain::create_swapchain(
        &device,
        &swapchain_loader,
        surface,
        &surface_caps,
        swapchain_format,
        window.inner_size().width,
        window.inner_size().height,
        graphics_family_index,
        vk::SwapchainKHR::null(),
    );

    let command_pool = swain::create_command_pool(&device, graphics_family_index);

    let command_buffers = swain::allocate_command_buffers(&device, command_pool, 2);

    let acquire_semaphore = swain::create_semaphore(&device);
    let submit_semaphore = swain::create_semaphore(&device);

    let command_buffer_fences = [swain::create_fence(&device), swain::create_fence(&device)];

    let render_pass = swain::create_render_pass(&device, swapchain_format);

    let mut framebuffers: Vec<vk::Framebuffer> = Vec::with_capacity(swapchain.image_views.len());
    for image_view in &swapchain.image_views {
        let framebuffer = swain::create_framebuffer(
            &device,
            *image_view,
            render_pass,
            swapchain.width,
            swapchain.height,
        );

        framebuffers.push(framebuffer);
    }

    let vertex_shader = swain::load_shader(&device, "data\\shaders\\triangle.vert.spv");
    let fragment_shader = swain::load_shader(&device, "data\\shaders\\triangle.frag.spv");

    let pipeline_layout = swain::create_pipeline_layout(&device);

    let graphics_pipeline = swain::create_graphics_pipeline(
        &device,
        vertex_shader,
        fragment_shader,
        pipeline_layout,
        render_pass,
    );

    let mut frame_index = 0;
    unsafe {
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent {
                    event:
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::MainEventsCleared => {
                    if window.inner_size().width > 0 && window.inner_size().height > 0 {
                        if swain::resize_swapchain_if_changed(
                            &mut swapchain,
                            &device,
                            &swapchain_loader,
                            surface,
                            &surface_caps,
                            swapchain_format,
                            window.inner_size().width,
                            window.inner_size().height,
                            graphics_family_index,
                        ) {
                            for i in 0..framebuffers.len() {
                                device.destroy_framebuffer(framebuffers[i], None);

                                framebuffers[i] = swain::create_framebuffer(
                                    &device,
                                    swapchain.image_views[i],
                                    render_pass,
                                    swapchain.width,
                                    swapchain.height,
                                );
                            }
                        }

                        device
                            .wait_for_fences(
                                &[command_buffer_fences[frame_index]],
                                true,
                                std::u64::MAX,
                            )
                            .unwrap();
                        device
                            .reset_fences(&[command_buffer_fences[frame_index]])
                            .unwrap();

                        let image_index = swapchain_loader
                            .acquire_next_image(
                                swapchain.swapchain,
                                std::u64::MAX,
                                acquire_semaphore,
                                vk::Fence::null(),
                            )
                            .unwrap()
                            .0;

                        let command_begin_info = vk::CommandBufferBeginInfo {
                            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                            ..Default::default()
                        };

                        device
                            .begin_command_buffer(command_buffers[frame_index], &command_begin_info)
                            .unwrap();

                        let render_begin_barrier = vk::ImageMemoryBarrier {
                            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                            old_layout: vk::ImageLayout::UNDEFINED,
                            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image: swapchain.images[image_index as usize],
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                level_count: 1,
                                layer_count: 1,
                                ..Default::default()
                            },
                            ..Default::default()
                        };
                        device.cmd_pipeline_barrier(
                            command_buffers[frame_index],
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            vk::DependencyFlags::BY_REGION,
                            &[],
                            &[],
                            &[render_begin_barrier],
                        );

                        let clear_values = [vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.2, 0.4, 0.6, 0.0],
                            },
                        }];

                        let render_pass_begin_info = vk::RenderPassBeginInfo {
                            render_pass: render_pass,
                            framebuffer: framebuffers[image_index as usize],
                            render_area: vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: vk::Extent2D {
                                    width: swapchain.width,
                                    height: swapchain.height,
                                },
                            },
                            clear_value_count: clear_values.len() as u32,
                            p_clear_values: clear_values.as_ptr(),
                            ..Default::default()
                        };

                        device.cmd_begin_render_pass(
                            command_buffers[frame_index],
                            &render_pass_begin_info,
                            vk::SubpassContents::INLINE,
                        );

                        let viewport = vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: swapchain.width as f32,
                            height: swapchain.height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        };
                        let scissor = vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: swapchain.width,
                                height: swapchain.height,
                            },
                        };

                        device.cmd_set_viewport(command_buffers[frame_index], 0, &[viewport]);
                        device.cmd_set_scissor(command_buffers[frame_index], 0, &[scissor]);

                        device.cmd_bind_pipeline(
                            command_buffers[frame_index],
                            vk::PipelineBindPoint::GRAPHICS,
                            graphics_pipeline,
                        );
                        device.cmd_draw(command_buffers[frame_index], 3, 1, 0, 0);

                        device.cmd_end_render_pass(command_buffers[frame_index]);

                        let render_end_barrier = vk::ImageMemoryBarrier {
                            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image: swapchain.images[image_index as usize],
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                level_count: 1,
                                layer_count: 1,
                                ..Default::default()
                            },
                            ..Default::default()
                        };
                        device.cmd_pipeline_barrier(
                            command_buffers[frame_index],
                            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            vk::DependencyFlags::BY_REGION,
                            &[],
                            &[],
                            &[render_end_barrier],
                        );

                        device
                            .end_command_buffer(command_buffers[frame_index])
                            .unwrap();

                        let submit_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
                        let submit_info = [vk::SubmitInfo {
                            wait_semaphore_count: 1,
                            p_wait_semaphores: &acquire_semaphore,
                            p_wait_dst_stage_mask: &submit_stage_mask,
                            command_buffer_count: 1,
                            p_command_buffers: &command_buffers[frame_index],
                            signal_semaphore_count: 1,
                            p_signal_semaphores: &submit_semaphore,
                            ..Default::default()
                        }];
                        device
                            .queue_submit(
                                graphics_queue,
                                &submit_info,
                                command_buffer_fences[frame_index],
                            )
                            .unwrap();

                        let present_info = vk::PresentInfoKHR {
                            wait_semaphore_count: 1,
                            p_wait_semaphores: &submit_semaphore,
                            swapchain_count: 1,
                            p_swapchains: &swapchain.swapchain,
                            p_image_indices: &image_index,
                            ..Default::default()
                        };

                        swapchain_loader
                            .queue_present(graphics_queue, &present_info)
                            .unwrap();

                        frame_index = (frame_index + 1) % 2;
                    }
                }
                _ => (),
            }
        });
    }
}
