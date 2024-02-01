use ash::extensions::khr;
use ash::vk;

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

pub struct Presentation {
    pub surface: vk::SurfaceKHR,
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

        let images = swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("Can't get Vulkan swapchain images.");

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

        device
            .create_image_view(&image_view_create_info, None)
            .expect("Can't create Vulkan image view.")
    }
}
