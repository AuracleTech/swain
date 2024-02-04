use log::warn;
use std::sync::{Arc, Mutex};
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

    let drawing = Arc::new(Mutex::new(true));
    let drawing_clone = Arc::clone(&drawing);

    let _handle = std::thread::spawn(move || loop {
        if *drawing_clone.lock().unwrap() {
            unsafe {
                engine.draw();
            }
        }

        let elapsed = engine.last_frame_present_time.elapsed();
        if elapsed > *swain::DRAW_TIME_MAX {
            warn!(
                "late by {} ms, restating frame immediately",
                elapsed.as_millis()
            );
        } else {
            let sleep_duration = *swain::DRAW_TIME_MAX - elapsed;
            std::thread::sleep(sleep_duration);
        }
    });

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
            Event::RedrawRequested(_) => {
                // let mut lock = drawing.lock().unwrap();
                // *lock = false;
            }
            _ => (),
        }
    })
}
