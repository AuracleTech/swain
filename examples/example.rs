const WIN_TITLE: &'static str = env!("CARGO_PKG_NAME");
const WIN_INIT_WIDTH: u32 = 1280;
const WIN_INIT_HEIGHT: u32 = 720;

fn main() {
    let _engine = swain::engine(WIN_TITLE, WIN_INIT_WIDTH, WIN_INIT_HEIGHT);
}
