use ash::extensions::khr;
use ash::vk;

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub depth: Depth,

    pub draw_commands_reuse_fence: vk::Fence,
    pub setup_commands_reuse_fence: vk::Fence,
}

pub struct Depth {
    pub format: vk::Format,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
}

pub struct Presentation {
    pub surface_khr: vk::SurfaceKHR,
    pub surface_caps: vk::SurfaceCapabilitiesKHR,
    pub format: vk::Format,
    pub queue_family_index: u32,
    pub swapchain_loader: khr::Swapchain,
    pub swapchain: Swapchain,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub framebuffers: Vec<vk::Framebuffer>,
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
        let swapchain_loader = khr::Swapchain::new(&instance, &device);

        let format = get_swapchain_format(physical_device, surface, &surface_loader);

        let swapchain = create_swapchain(
            instance,
            physical_device,
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
            surface_khr: surface,
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
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
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

        // SECTION : Chroma & Image Views
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

        let images = swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("Can't get Vulkan swapchain images.");

        let mut image_views: Vec<vk::ImageView> = Vec::with_capacity(images.len());
        for image in &images {
            let image_view_create_info = vk::ImageViewCreateInfo {
                image: *image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
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

            image_views.push(image_view);
        }

        // SECTION : Depth image & view
        let depth_format = vk::Format::D32_SFLOAT;
        let device_memory_properties =
            instance.get_physical_device_memory_properties(physical_device);
        let depth_image_info = vk::ImageCreateInfo {
            format: depth_format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let depth_image = device
            .create_image(&depth_image_info, None)
            .expect("Failed to create depth image.");

        let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
        let depth_image_memory_index = find_memorytype_index(
            &depth_image_memory_req,
            &device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("Unable to find suitable memory type index for depth image.");

        let depth_image_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: depth_image_memory_req.size,
            memory_type_index: depth_image_memory_index,
            ..Default::default()
        };
        let depth_image_memory = device
            .allocate_memory(&depth_image_allocate_info, None)
            .expect("Failed to allocate depth image memory.");
        device
            .bind_image_memory(depth_image, depth_image_memory, 0)
            .expect("Failed to bind depth image memory.");

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

        let fence_create_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        let draw_commands_reuse_fence = device
            .create_fence(&fence_create_info, None)
            .expect("Create fence failed.");
        let setup_commands_reuse_fence = device
            .create_fence(&fence_create_info, None)
            .expect("Create fence failed.");

        Swapchain {
            swapchain,
            images,
            image_views,
            depth: Depth {
                format: depth_format,
                image: depth_image,
                image_view: depth_image_view,
            },
            draw_commands_reuse_fence,
            setup_commands_reuse_fence,
        }
    }
}

pub fn find_memorytype_index(
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
