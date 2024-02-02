use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
};

const APP_VERSION: u32 = 30012024;
const WIN_TITLE: &'static str = env!("CARGO_PKG_NAME");
const WIN_INIT_WIDTH: u32 = 512;
const WIN_INIT_HEIGHT: u32 = 512;

fn main() {
    let (mut engine, event_loop) =
        swain::Engine::new(APP_VERSION, WIN_TITLE, WIN_INIT_WIDTH, WIN_INIT_HEIGHT);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => (),
            },
            // Event::MainEventsCleared => engine.window.request_redraw(),
            Event::RedrawRequested(_) => engine.draw(),
            _ => (),
        }
    });
}
