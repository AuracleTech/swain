const WIN_TITLE: &'static str = "Vulkan";
const WIN_INIT_WIDTH: u32 = 1600;
const WIN_INIT_HEIGHT: u32 = 900;

fn main() {
    let _engine = swain::engine(WIN_TITLE, WIN_INIT_WIDTH, WIN_INIT_HEIGHT);
}
